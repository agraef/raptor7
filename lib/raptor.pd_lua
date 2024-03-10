
-- Raptor Random Arpeggiator V7

-- This is a backport of the Ardour version of the same plugin at
-- https://github.com/agraef/ardour-lua, included in Ardour 8.0 and later,
-- which in turn is based on https://github.com/agraef/raptor-lua. It has a
-- simplified codebase and offers some new and improved features such as latch
-- mode. Therefore a new Pd version of the plugin was in order, which is the
-- version you're looking at.

-- Author: Albert Gräf <aggraef@gmail.com>, Dept. of Music-Informatics,
-- Johannes Gutenberg University (JGU) of Mainz, Germany, please check
-- https://agraef.github.io/ for a list of my software.

-- Copyright (c) 2021, 2024 by Albert Gräf <aggraef@gmail.com>

-- Distributed under the GPLv3+, please check the accompanying COPYING file
-- for details.

-- -------------------------------------------------------------------------

local raptor = pd.Class:new():register("raptor")

-- print is used for debugging purposes, output goes to the Pd console

local function print(...)
   local s = ""
   for i,v in ipairs{...} do
      if i == 1 then
	 s = tostring(v)
      else
	 s = s .. " " .. tostring(v)
      end
   end
   pd.post(s)
end

-- -------------------------------------------------------------------------

-- load kikito's inspect; we mostly need this for debugging messages, but also
-- when saving data, so the output doesn't need to be pretty, but should be
-- humanly readable and conform to Lua syntax

local _inspect = require('inspect')

-- adjust the formatting options
local function inspect(x)
   return _inspect(x, {newline = " ", indent = ""})
end

-- -------------------------------------------------------------------------

-- Various helper functions to compute Barlow meters and harmonicities using
-- the methods from Clarence Barlow's Ratio book (Feedback Papers, Cologne,
-- 2001)


local M = {}

-- list helper functions

-- concatenate tables
function M.tableconcat(t1, t2)
   local res = {}
   for i=1,#t1 do
      table.insert(res, t1[i])
   end
   for i=1,#t2 do
      table.insert(res, t2[i])
   end
   return res
end

-- reverse a table
function M.reverse(list)
   local res = {}
   for _, v in ipairs(list) do
      table.insert(res, 1, v)
   end
   return res
end

-- arithmetic sequences
function M.seq(from, to, step)
   step = step or 1;
   local sgn = step>=0 and 1 or -1
   local res = {}
   while sgn*(to-from) >= 0 do
      table.insert(res, from)
      from = from + step
   end
   return res
end

-- cycle through a table
function M.cycle(t, i)
   local n = #t
   if n > 0 then
      while i > n do
	 i = i - n
      end
   end
   return t[i]
end

-- some functional programming goodies

function M.map(list, fn)
   local res = {}
   for _, v in ipairs(list) do
      table.insert(res, fn(v))
   end
   return res
end

function M.reduce(list, acc, fn)
   for _, v in ipairs(list) do
      acc = fn(acc, v)
   end
   return acc
end

function M.collect(list, acc, fn)
   local res = {acc}
   for _, v in ipairs(list) do
      acc = fn(acc, v)
      table.insert(res, acc)
   end
   return res
end

function M.sum(list)
   return M.reduce(list, 0, function(a,b) return a+b end)
end

function M.prd(list)
   return M.reduce(list, 1, function(a,b) return a*b end)
end

function M.sums(list)
   return M.collect(list, 0, function(a,b) return a+b end)
end

function M.prds(list)
   return M.collect(list, 1, function(a,b) return a*b end)
end

-- Determine the prime factors of an integer. The result is a list with the
-- prime factors in non-decreasing order.

function M.factor(n)
   local factors = {}
   if n<0 then n = -n end
   while n % 2 == 0 do
      table.insert(factors, 2)
      n = math.floor(n / 2)
   end
   local p = 3
   while p <= math.sqrt(n) do
      while n % p == 0 do
	 table.insert(factors, p)
	 n = math.floor(n / p)
      end
      p = p + 2
   end
   if n > 1 then -- n must be prime
      table.insert(factors, n)
   end
   return factors
end

-- Collect the factors of the integer n and return them as a list of pairs
-- {p,k} where p are the prime factors in ascending order and k the
-- corresponding (nonzero) multiplicities. If the given number is a pair {p,
-- q}, considers p/q as a rational number and returns its prime factors with
-- positive or negative multiplicities.

function M.factors(x)
   if type(x) == "table" then
      local n, m = table.unpack(x)
      local pfs, nfs, mfs = {}, M.factors(n), M.factors(m)
      -- merge the factors in nfs and mfs into a single list
      local i, j, k, N, M = 1, 1, 1, #nfs, #mfs
      while i<=N or j<=M do
	 if j>M or (i<=N and mfs[j][1]>nfs[i][1]) then
	    pfs[k] = nfs[i]
	    k = k+1; i = i+1
	 elseif i>N or (j<=M and nfs[i][1]>mfs[j][1]) then
	    pfs[k] = mfs[j]
	    pfs[k][2] = -mfs[j][2]
	    k = k+1; j = j+1
	 else
	    pfs[k] = nfs[i]
	    pfs[k][2] = nfs[i][2] - mfs[j][2]
	    k = k+1; i = i+1; j = j+1
	 end
      end
      return pfs
   else
      local pfs, pf = {}, M.factor(x)
      if next(pf) then
	 local j, n = 1, #pf
	 pfs[j] = {pf[1], 1}
	 for i = 2, n do
	    if pf[i] == pfs[j][1] then
	       pfs[j][2] = pfs[j][2] + 1
	    else
	       j = j+1
	       pfs[j] = {pf[i], 1}
	    end
	 end
      end
      return pfs
   end
end

-- Probability functions. These are used with some of the random generation
-- functions below.

-- Create random permutations. Chooses n random values from a list ms of input
-- values according to a probability distribution given by a list ws of
-- weights. NOTES: ms and ws should be of the same size, otherwise excess
-- elements will be chosen at random. In particular, if ws is empty or missing
-- then shuffle(n, ms) will simply return n elements chosen from ms at random
-- using a uniform distribution. ms and ws and are modified *in place*,
-- removing chosen elements, so that their final contents will be the elements
-- *not* chosen and their corresponding weight distribution.

function M.shuffle(n, ms, ws)
   local res = {}
   if ws == nil then
      -- simply choose elements at random, uniform distribution
      ws = {}
   end
   while next(ms) ~= nil and n>0 do
      -- accumulate weights
      local sws = M.sums(ws)
      local s = sws[#sws]
      table.remove(sws, 1)
      -- pick a random index
      local k, r = 0, math.random()*s
      --print("r = ", r, "sws = ", table.unpack(sws))
      for i = 1, #sws do
	 if r < sws[i] then
	    k = i; break
	 end
      end
      -- k may be out of range if ws and ms aren't of the same size, in which
      -- case we simply pick an element at random
      if k==0 or k>#ms then
	 k = math.random(#ms)
      end
      table.insert(res, ms[k])
      n = n-1; table.remove(ms, k);
      if k<=#ws then
	 table.remove(ws, k)
      end
   end
   return res
end

-- Calculate modulated values. This is used for all kinds of parameters which
-- can vary automatically according to pulse strength, such as note
-- probability, velocity, gate, etc.

function M.mod_value(x1, x2, b, w)
   -- x2 is the nominal value which is always output if b==0. As b increases
   -- or decreases, the range extends downwards towards x1. (Normally,
   -- x2>x1, but you can reverse bounds to have the range extend upwards.)
   if b >= 0 then
      -- positive bias: mod_value(w) -> x1 as w->0, -> x2 as w->1
      -- zero bias: mod_value(w) == x2 (const.)
      return x2-b*(1-w)*(x2-x1)
   else
      -- negative bias: mod_value(w) -> x1 as w->1, -> x2 as w->0
      return x2+b*w*(x2-x1)
   end
end

-- Barlow meters. This stuff is mostly a verbatim copy of the guts of
-- meter.pd_lua, please check that module for details.

-- Computes the best subdivision q in the range 1..n and pulse p in the range
-- 0..q so that p/q matches the given phase f in the floating point range 0..1
-- as closely as possible. Returns p, q and the absolute difference between f
-- and p/q. NB: Seems to work best for q values up to 7.

function M.subdiv(n, f)
   local best_p, best_q, best = 0, 0, 1
   for q = 1, n do
      local p = math.floor(f*q+0.5) -- round towards nearest pulse
      local diff = math.abs(f-p/q)
      if diff < best then
	 best_p, best_q, best = p, q, diff
      end
   end
   return best_p, best_q, best
end

-- Compute pulse strengths according to Barlow's indispensability formula from
-- the Ratio book.

function M.indisp(q)
   local function ind(q, k)
      -- prime indispensabilities
      local function pind(q, k)
	 local function ind1(q, k)
	    local i = ind(M.reverse(M.factor(q-1)), k)
	    local j = i >= math.floor(q / 4) and 1 or 0;
	    return i+j
	 end
	 if q <= 3 then
	    return (k-1) % q
	 elseif k == q-2 then
	    return math.floor(q / 4)
	 elseif k == q-1 then
	    return ind1(q, k-1)
	 else
	    return ind1(q, k)
	 end
      end
      local s = M.prds(q)
      local t = M.reverse(M.prds(M.reverse(q)))
      return
	 M.sum(M.map(M.seq(1, #q), function(i) return s[i] * pind(q[i], (math.floor((k-1) % t[1] / t[i+1]) + 1) % q[i]) end))
   end
   if type(q) == "number" then
      q = M.factor(q)
   end
   if type(q) ~= "table" then
      error("invalid argument, must be an integer or table of primes")
   else
      return M.map(M.seq(0,M.prd(q)-1), function(k) return ind(q,k) end)
   end
end

-- Barlow harmonicities from the Ratio book. These are mostly ripped out of an
-- earlier version of the Raptor random arpeggiator programs (first written in
-- Q, then rewritten in Pure, and now finally ported to Lua).

-- Some "standard" 12 tone scales and prime valuation functions to play with.
-- Add others as needed. We mostly use the just scale and the standard Barlow
-- valuation here.

M.just = -- standard just intonation, a.k.a. the Ptolemaic (or Didymic) scale
   {  {1,1}, {16,15}, {9,8}, {6,5}, {5,4}, {4,3}, {45,32},
      {3,2}, {8,5}, {5,3}, {16,9}, {15,8}, {2,1}  }
M.pyth = -- pythagorean (3-limit) scale
   {  {1,1}, {2187,2048}, {9,8}, {32,27}, {81,64}, {4,3}, {729,512},
      {3,2}, {6561,4096}, {27,16}, {16,9}, {243,128}, {2,1}  }
M.mean4 = -- 1/4 comma meantone scale, Barlow (re-)rationalization
   {  {1,1}, {25,24}, {10,9}, {6,5}, {5,4}, {4,3}, {25,18},
      {3,2}, {25,16}, {5,3}, {16,9}, {15,8}, {2,1}  }

function M.barlow(p)	return 2*(p-1)*(p-1)/p end
function M.euler(p)	return p-1 end
-- "mod 2" versions (octave is eliminated)
function M.barlow2(p)	if p==2 then return 0 else return M.barlow(p) end end
function M.euler2(p)	if p==2 then return 0 else return M.euler(p) end end

-- Harmonicity computation.

-- hrm({p,q}, pv) computes the disharmonicity of the interval p/q using the
-- prime valuation function pv.

-- hrm_dist({p1,q1}, {p2,q2}, pv) computes the harmonic distance between two
-- pitches, i.e., the disharmonicity of the interval between {p1,q1} and
-- {p2,q2}.

-- hrm_scale(S, pv) computes the disharmonicity metric of a scale S, i.e., the
-- pairwise disharmonicities of all intervals in the scale. The input is a
-- list of intervals as {p,q} pairs, the output is the distance matrix.

function M.hrm(x, pv)
   return M.sum(M.map(M.factors(x),
	function(f) local p, k = table.unpack(f)
	   return math.abs(k) * pv(p)
	end))
end

function M.hrm_dist(x, y, pv)
   local p1, q1 = table.unpack(x)
   local p2, q2 = table.unpack(y)
   return M.hrm({p1*q2,p2*q1}, pv)
end

function M.hrm_scale(S, pv)
   return M.map(S,
	function(s)
	   return M.map(S, function(t) return M.hrm_dist(s, t, pv) end)
	end)
end

-- Some common tables for convenience and testing. These are all based on a
-- standard 12-tone just tuning. NOTE: The given reference tables use rounded
-- values, but are good enough for most practical purposes; you might want to
-- employ these to avoid the calculation cost.

-- Barlow's "indigestibility" harmonicity metric
-- M.bgrad = {0,13.07,8.33,10.07,8.4,4.67,16.73,3.67,9.4,9.07,9.33,12.07,1}
M.bgrad = M.map(M.just, function(x) return M.hrm(x, M.barlow) end)

-- Euler's "gradus suavitatis" (0-based variant)
-- M.egrad = {0,10,7,7,6,4,13,3,7,6,8,9,1}
M.egrad = M.map(M.just, function(x) return M.hrm(x, M.euler) end)

-- In an arpeggiator we might want to treat different octaves of the same
-- pitch as equivalent, in which case we can use the following "mod 2" tables:
M.bgrad2 = M.map(M.just, function(x) return M.hrm(x, M.barlow2) end)
M.egrad2 = M.map(M.just, function(x) return M.hrm(x, M.euler2) end)

-- But in the following we stick to the standard Barlow table.
M.grad = M.bgrad

-- Calculate the harmonicity of the interval between two (MIDI) notes.
function M.hm(n, m)
   local d = math.max(n, m) - math.min(n, m)
   return 1/(1+M.grad[d%12+1])
end

-- Use this instead if you also want to keep account of octaves.
function M.hm2(n, m)
   local d = math.max(n, m) - math.min(n, m)
   return 1/(1+M.grad[d%12+1]+(d//12)*M.grad[13])
end

-- Calculate the average harmonicity (geometric mean) of a MIDI note relative
-- to a given chord (specified as a list of MIDI notes).
function M.hv(ns, m)
   if next(ns) ~= nil then
      local xs = M.map(ns, function(n) return M.hm(m, n) end)
      return M.prd(xs)^(1/#xs)
   else
      return 1
   end
end

-- Sort the MIDI notes in ms according to descending average harmonicities
-- w.r.t. the MIDI notes in ns. This allows you to quickly pick the "best"
-- (harmonically most pleasing) MIDI notes among given alternatives ms
-- w.r.t. a given chord ns.
function M.besthv(ns, ms)
   local mhv = M.map(ms, function(m) return {m, M.hv(ns, m)} end)
   table.sort(mhv, function(x, y) return x[2]>y[2] or
		 (x[2]==y[2] and x[1]<y[1]) end)
   return M.map(mhv, function(x) return x[1] end)
end

-- Randomized note filter. This is the author's (in)famous Raptor algorithm.
-- It needs a whole bunch of parameters, but also delivers much more
-- interesting results and can produce randomized chords as well. Basically,
-- it performs a random walk guided by Barlow harmonicities and
-- indispensabilities. The parameters are:

-- ns: input notes (chord memory of the arpeggiator, as in besthv these are
-- used to calculate the average harmonicities)

-- ms: candidate output notes (these will be filtered and participate in the
-- random walk)

-- w: indispensability value used to modulate the various parameters

-- nmax, nmod: range and modulation of the density (maximum number of notes
-- in each step)

-- smin, smax, smod: range and modulation of step widths, which limits the
-- steps between notes in successive pulses

-- dir, mode, uniq: arpeggio direction (0 = random, 1 = up, -1 = down), mode
-- (0 = random, 1 = up, 2 = down, 3 = up-down, 4 = down-up), and whether
-- repeated notes are disabled (uniq flag)

-- hmin, hmax, hmod: range and modulation of eligible harmonicities, which are
-- used to filter candidate notes based on average harmonicities w.r.t. the
-- input notes

-- pref, prefmod: range and modulation of harmonic preference. This is
-- actually one of the most important and effective parameters in the Raptor
-- algorithm which drives the random note selection process. A pref value
-- between -1 and 1 determines the weighted probabilities used to pick notes
-- at random. pref>0 gives preference to notes with high harmonicity, pref<0
-- to notes with low harmonicity, and pref==0 ignores harmonicity (in which
-- case all eligible notes are chosen with the same probability). The prefs
-- parameter can also be modulated by pulse strengths as indicated by prefmod
-- (prefmod>0 lowers preference on weak pulses, prefmod<0 on strong pulses).

function M.harm_filter(w, hmin, hmax, hmod, ns, ms)
   -- filters notes according to harmonicities and a given pulse weight w
   if next(ns) == nil then
      -- empty input (no eligible notes)
      return {}
   else
      local res = {}
      for _,m in ipairs(ms) do
	 local h = M.hv(ns, m)
	 -- modulate: apply a bias determined from hmod and w
	 if hmod > 0 then
	    h = h^(1-hmod*(1-w))
	 elseif hmod < 0 then
	    h = h^(1+hmod*w)
	 end
	 -- check that the (modulated) harmonicity is within prescribed bounds
	 if h>=hmin and h<=hmax then
	    table.insert(res, m)
	 end
      end
      return res
   end
end

function M.step_filter(w, smin, smax, smod, dir, mode, cache, ms)
   -- filters notes according to the step width parameters and pulse weight w,
   -- given which notes are currently playing (the cache)
   if next(ms) == nil or dir == 0 then
      return ms, dir
   end
   local res = {}
   while next(res) == nil do
      if next(cache) ~= nil then
	 -- non-empty cache, going any direction
	 local lo, hi = cache[1], cache[#cache]
	 -- NOTE: smin can be negative, allowing us, say, to actually take a
	 -- step *down* while going upwards. But we always enforce that smax
	 -- is non-negative in order to avoid deadlock situations where *no*
	 -- step is valid anymore, and even restarting the pattern doesn't
	 -- help. (At least that's what I think, I don't really recall what
	 -- the original rationale behind all this was, but since it's in the
	 -- original Raptor code, it must make sense somehow. ;-)
	 smax = math.max(0, smax)
	 smax = math.floor(M.mod_value(math.abs(smin), smax, smod, w)+0.5)
	 local function valid_step_min(m)
	    if dir==0 then
	       return (m>=lo+smin) or (m<=hi-smin)
	    elseif dir>0 then
	       return m>=lo+smin
	    else
	       return m<=hi-smin
	    end
	 end
	 local function valid_step_max(m)
	    if dir==0 then
	       return (m>=lo-smax) and (m<=hi+smax)
	    elseif dir>0 then
	       return (m>=lo+math.min(0,smin)) and (m<=hi+smax)
	    else
	       return (m>=lo-smax) and (m<=hi-math.min(0,smin))
	    end
	 end
	 for _,m in ipairs(ms) do
	    if valid_step_min(m) and valid_step_max(m) then
	       table.insert(res, m)
	    end
	 end
      elseif dir == 1 then
	 -- empty cache, going up, start at bottom
	 local lo = ms[1]
	 local max = math.floor(M.mod_value(smin, smax, smod, w)+0.5)
	 for _,m in ipairs(ms) do
	    if m <= lo+max then
	       table.insert(res, m)
	    end
	 end
      elseif dir == -1 then
	 -- empty cache, going down, start at top
	 local hi = ms[#ms]
	 local max = math.floor(M.mod_value(smin, smax, smod, w)+0.5)
	 for _,m in ipairs(ms) do
	    if m >= hi-max then
	       table.insert(res, m)
	    end
	 end
      else
	 -- empty cache, random direction, all notes are eligible
	 return ms, dir
      end
      if next(res) == nil then
	 -- we ran out of notes, restart the pattern
	 -- print("raptor: no notes to play, restart!")
	 cache = {}
	 if mode==0 then
	    dir = 0
	 elseif mode==1 or (mode==3 and dir==0) then
	    dir = 1
	 elseif mode==2 or (mode==4 and dir==0) then
	    dir = -1
	 else
	    dir = -dir
	 end
      end
   end
   return res, dir
end

function M.uniq_filter(uniq, cache, ms)
   -- filters out repeated notes (removing notes already in the cache),
   -- depending on the uniq flag
   if not uniq or next(ms) == nil or next(cache) == nil then
      return ms
   end
   local res = {}
   local i, j, k, N, M = 1, 1, 1, #cache, #ms
   while i<=N or j<=M do
      if j>M then
	 -- all elements checked, we're done
	 return res
      elseif i>N or ms[j]<cache[i] then
	 -- current element not in cache, add it
	 res[k] = ms[j]
	 k = k+1; j = j+1
      elseif ms[j]>cache[i] then
	 -- look at next cache element
	 i = i+1
      else
	 -- current element in cache, skip it
	 i = i+1; j = j+1
      end
   end
   return res
end

function M.pick_notes(w, n, pref, prefmod, ns, ms)
   -- pick n notes from the list ms of eligible notes according to the
   -- given harmonic preference
   local ws = {}
   -- calculate weighted harmonicities based on preference; this gives us the
   -- probability distribution for the note selection step
   local p = M.mod_value(0, pref, prefmod, w)
   if p==0 then
      -- no preference, use uniform distribution
      for i = 1, #ms do
	 ws[i] = 1
      end
   else
      for i = 1, #ms do
	 -- "Frankly, I don't know where the exponent came from," probably
	 -- experimentation. ;-)
	 ws[i] = M.hv(ns, ms[i]) ^ (p*10)
      end
   end
   return M.shuffle(n, ms, ws)
end

-- The note generator. This is invoked with the current pulse weight w, the
-- current cache (notes played in the previous step), the input notes ns, the
-- candidate output notes ms, and all the other parameters that we need
-- (density: nmax, nmod; harmonicity: hmin, hmax, hmod; step width: smin,
-- smax, smod; arpeggiator state: dir, mode, uniq; harmonic preference: pref,
-- prefmod). It returns a selection of notes chosen at random for the given
-- parameters, along with the updated direction dir of the arpeggiator.

function M.rand_notes(w, nmax, nmod,
		      hmin, hmax, hmod,
		      smin, smax, smod,
		      dir, mode, uniq,
		      pref, prefmod,
		      cache,
		      ns, ms)
   -- uniqueness filter: remove repeated notes
   local res = M.uniq_filter(uniq, cache, ms)
   -- harmonicity filter: select notes based on harmonicity
   res = M.harm_filter(w, hmin, hmax, hmod, ns, res)
   -- step filter: select notes based on step widths and arpeggiator state
   -- (this must be the last filter!)
   res, dir = M.step_filter(w, smin, smax, smod, dir, mode, cache, res)
   -- pick notes
   local n = math.floor(M.mod_value(1, nmax, nmod, w)+0.5)
   res = M.pick_notes(w, n, pref, prefmod, ns, res)
   return res, dir
end

local barlow = M

-- -------------------------------------------------------------------------

-- Arpeggiator object.

arpeggio = {}
arpeggio.__index = arpeggio

function arpeggio:new(m) -- constructor
   local x = setmetatable(
      {
	 -- some reasonable defaults (see also arpeggio:initialize below)
	 debug = 0, idx = 0, chord = {}, pattern = {},
	 latch = nil, down = -1, up = 1, mode = 0,
	 minvel = 60, maxvel = 120, velmod = 1,
	 wmin = 0, wmax = 1,
	 pmin = 0.3, pmax = 1, pmod = 0,
	 gate = 1, gatemod = 0,
	 veltracker = 1, minavg = nil, maxavg = nil,
	 gain = 1, g =  math.exp(-1/3),
	 loopstate = 0, loopsize = 0, loopidx = 0, loop = {}, loopdir = "",
	 nmax = 1, nmod = 0,
	 hmin = 0, hmax = 1, hmod = 0,
	 smin = 1, smax = 7, smod = 0,
	 uniq = 1,
	 pref = 1, prefmod = 0,
	 pitchtracker = 0, pitchlo = 0, pitchhi = 0,
	 n = 0
      },
      arpeggio)
   x:initialize(m)
   return x
end

function arpeggio:initialize(m)
   -- debugging (bitmask): 1 = pattern, 2 = input, 4 = output
   self.debug = 0
   -- internal state variables
   self.idx = 0
   self.chord = {}
   self.pattern = {}
   self.latch = nil
   self.down, self.up, self.mode = -1, 1, 0
   self.minvel, self.maxvel, self.velmod = 60, 120, 1
   self.pmin, self.pmax, self.pmod = 0.3, 1, 0
   self.wmin, self.wmax = 0, 1
   self.gate, self.gatemod = 1, 0
   -- velocity tracker
   self.veltracker, self.minavg, self.maxavg = 1, nil, nil
   -- This isn't really a "gain" control any more, it's more like a dry/wet
   -- mix (1 = dry, 0 = wet) between set values (minvel, maxvel) and the
   -- calculated envelope of MIDI input notes (minavg, maxavg).
   self.gain = 1
   -- smoothing filter, time in pulses (3 works for me, YMMV)
   local t = 3
    -- filter coefficient
   self.g = math.exp(-1/t)
   -- looper
   self.loopstate = 0
   self.loopsize = 0
   self.loopidx = 0
   self.loop = {}
   self.loopdir = ""
   -- Raptor params, reasonable defaults
   self.nmax, self.nmod = 1, 0
   self.hmin, self.hmax, self.hmod = 0, 1, 0
   self.smin, self.smax, self.smod = 1, 7, 0
   self.uniq = 1
   self.pref, self.prefmod = 1, 0
   self.pitchtracker = 0
   self.pitchlo, self.pitchhi = 0, 0
   -- Barlow meter
   -- XXXTODO: We only do integer pulses currently, so the subdivisions
   -- parameter self.n is currently disabled. Maybe we can find some good use
   -- for it in the future, e.g., for ratchets?
   self.n = 0
   if m == nil then
      m = {4} -- default meter (common time)
   end
   -- initialize the indispensability tables and reset the beat counter
   self.indisp = {}
   self:prepare_meter(m)
   -- return the initial number of beats
   return self.beats
end

-- Barlow indispensability meter computation, cf. barlow.pd_lua. This takes a
-- zero-based beat number, optionally with a phase in the fractional part to
-- indicate a sub-pulse below the beat level. We then compute the closest
-- matching subdivision and compute the corresponding pulse weight, using the
-- precomputed indispensability tables. The returned result is a pair w,n
-- denoting the Barlow indispensability weight of the pulse in the range
-- 0..n-1, where n denotes the total number of beats (number of beats in the
-- current meter times the current subdivision).

-- list helpers
local tabcat, reverse, cycle, map, seq = barlow.tableconcat, barlow.reverse, barlow.cycle, barlow.map, barlow.seq
-- Barlow indispensabilities and friends
local factor, indisp, subdiv = barlow.factor, barlow.indisp, barlow.subdiv
-- Barlow harmonicities and friends
local mod_value, rand_notes = barlow.mod_value, barlow.rand_notes

function arpeggio:meter(b)
   if b < 0 then
      error("meter: beat index must be nonnegative")
      return
   end
   local beat, f = math.modf(b)
   -- take the beat index modulo the total number of beats
   beat = beat % self.beats
   if self.n > 0 then
      -- compute the closest subdivision for the given fractional phase
      local p, q = subdiv(self.n, f)
      if self.last_q then
	 local x = self.last_q / q
	 if math.floor(x) == x then
	    -- If the current best match divides the previous one, stick to
	    -- it, in order to prevent the algorithm from quickly changing
	    -- back to the root meter at each base pulse. XXFIXME: This may
	    -- stick around indefinitely until the meter changes. Maybe we'd
	    -- rather want to reset this automatically after some time (such
	    -- as a complete bar without non-zero phases)?
	    p, q = x*p, x*q
	 end
      end
      self.last_q = q
      -- The overall zero-based pulse index is beat*q + p. We add 1 to
      -- that to get a 1-based index into the indispensabilities table.
      local w = self.indisp[q][beat*q+p+1]
      return w, self.beats*q
   else
      -- no subdivisions, just return the indispensability and number of beats
      -- as is
      local w = self.indisp[1][beat+1]
      return w, self.beats
   end
end

function arpeggio:numarg(x)
   if type(x) == "table" then
      x = x[1]
   end
   if type(x) == "number" then
      return x
   else
      error("arpeggio: expected number, got " .. tostring(x))
   end
end

function arpeggio:intarg(x)
   if type(x) == "table" then
      x = x[1]
   end
   if type(x) == "number" then
      return math.floor(x)
   else
      error("arpeggio: expected integer, got " .. tostring(x))
   end
end

-- the looper

function arpeggio:loop_clear()
   -- reset the looper
   self.loopstate = 0
   self.loopidx = 0
   self.loop = {}
end

function arpeggio:loop_set()
   -- set the loop and start playing it
   local n, m = #self.loop, self.loopsize
   local b, p, q = self.beats, self.loopidx, self.idx
   -- NOTE: Use Ableton-style launch quantization here. We quantize start and
   -- end of the loop, as well as m = the target loop size to whole bars, to
   -- account for rhythmic inaccuracies. Otherwise it's just much too easy to
   -- miss bar boundaries when recording a loop.
   m = math.ceil(m/b)*b -- rounding up
   -- beginning of last complete bar in cyclic buffer
   local k = (p-q-b) % 256
   if n <= 0 or m <= 0 or m > 256 or k >= n then
      -- We haven't recorded enough steps for a bar yet, or the target size is
      -- 0, bail out with an empty loop.
      self.loop = {}
      self.loopidx = 0
      self.loopstate = 1
      if m == 0 then
	 print("loop: zero loop size")
      else
	 print(string.format("loop: got %d steps, need %d.", p>=n and math.max(0, p-q) or q==0 and n or math.max(0, n-b), b))
      end
      return
   end
   -- At this point we have at least 1 bar, starting at k+1, that we can grab;
   -- try extending the loop until we hit the target size.
   local l = b
   while l < m do
      if k >= b then
	 k = k-b
      elseif p >= n or (k-b) % 256 < p then
	 -- in this case either the cyclic buffer hasn't been filled yet, or
	 -- wrapping around would take us past the buffer pointer, so bail out
	 break
      else
	 -- wrap around to the end of the buffer
	 k = (k-b) % 256
      end
      l = l+b
   end
   -- grab l (at most m) steps
   --print(string.format("loop: recorded %d/%d steps %d-%d", l, m, k+1, k+m))
   print(string.format("loop: recorded %d/%d steps", l, m))
   local loop = {}
   for i = k+1, k+l do
      loop[i-k] = cycle(self.loop, i)
   end
   self.loop = loop
   self.loopidx = q % l
   self.loopstate = 1
end

function arpeggio:loop_add(notes, vel, gate)
   -- we only start recording at the first note
   local have_notes = type(notes) == "number" or
      (notes ~= nil and next(notes) ~= nil)
   if have_notes or next(self.loop) ~= nil then
      self.loop[self.loopidx+1] = {notes, vel, gate}
      -- we always *store* up to 256 steps in a cyclic buffer
      self.loopidx = (self.loopidx+1) % 256
   end
end

function arpeggio:loop_get()
   local res = {{}, 0, 0}
   local p, n = self.loopidx, math.min(#self.loop, self.loopsize)
   if p < n then
      res = self.loop[p+1]
      -- we always *read* exactly n steps in a cyclic buffer
      self.loopidx = (p+1) % n
      if p % self.beats == 0 then
	 local a, b = p // self.beats + 1, n // self.beats
	 print(string.format("loop: playing bar %d/%d", a, b))
      end
   end
   -- we maybe should return the current loopidx here which is used to give
   -- visual feedback about the loop cycle in the Pd external; not sure how to
   -- do this in Ardour, though
   return res
end

local function fexists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function arpeggio:loop_file(file, cmd)
   -- default for cmd is 1 (save) if loop is playing, 0 (load) otherwise
   cmd = cmd or self.loopstate
   -- apply the loopdir if any
   local path = self.loopdir .. file
   if cmd == 1 then
      -- save: first create a backup copy if the file already exists
      if fexists(path) then
	 local k, bakname = 1
	 repeat
	    bakname = string.format("%s~%d~", path, k)
	    k = k+1
	 until not fexists(bakname)
	 -- ignore errors, if we can't rename the file, we probably can't
	 -- overwrite it either
	 os.rename(path, bakname)
      end
      local f, err = io.open(path, "w")
      if type(err) == "string" then
	 print(string.format("loop: %s", err))
	 return
      end
      -- shorten the table to the current loop size if needed
      local loop, n = {}, math.min(#self.loop, self.loopsize)
      table.move(self.loop, 1, n, 1, loop)
      -- add some pretty-printing
      local function bars(level, count)
	 if level == 1 and count%self.beats == 0 then
	    return string.format("-- bar %d", count//self.beats+1)
	 end
      end
      f:write(string.format("-- saved by Raptor %s\n", os.date()))
      f:write(inspect(loop, {extra = 1, addin = bars}))
      f:close()
      print(string.format("loop: %s: saved %d steps", file, n))
   elseif cmd == 0 then
      -- load: check that file exists and is loadable
      local f, err = io.open(path, "r")
      if type(err) == "string" then
	 print(string.format("loop: %s", err))
	 return
      end
      local fun, err = load("return " .. f:read("a"))
      f:close()
      if type(err) == "string" or type(fun) ~= "function" then
	 print(string.format("loop: %s: invalid format", file))
      else
	 local loop = fun()
	 if type(loop) ~= "table" then
	    print(string.format("loop: %s: invalid format", file))
	 else
	    self.loop = loop
	    self.loopsize = #loop
	    self.loopidx = self.idx % math.max(1, self.loopsize)
	    self.loopstate = 1
	    print(string.format("loop: %s: loaded %d steps", file, #loop))
	    return self.loopsize
	 end
      end
   elseif cmd == 2 then
      -- check that file exists, report result
      return fexists(path) and 1 or 0
   end
end

function arpeggio:set_loopsize(x)
   x = self:intarg(x)
   if type(x) == "number" then
      self.loopsize = math.max(0, math.min(256, x))
      if self.loopstate == 1 then
	 -- need to update the loop index in case the loopsize changed
	 if self.loopsize > 0 then
	    -- also resynchronize the loop with the arpeggiator if needed
	    self.loopidx = math.max(self.idx, self.loopidx % self.loopsize)
	 else
	    self.loopidx = 0
	 end
      end
   end
end

function arpeggio:set_loop(x)
   if type(x) == "string" then
      x = {x}
   end
   if type(x) == "table" and type(x[1]) == "string" then
      -- file operations
      self:loop_file(table.unpack(x))
   else
      x = self:intarg(x)
      if type(x) == "number" then
	 if x ~= 0 and self.loopstate == 0 then
	    self:loop_set()
	 elseif x == 0 and self.loopstate == 1 then
	    self:loop_clear()
	 end
      end
   end
end

function arpeggio:set_loopdir(x)
   if type(x) == "string" then
      x = {x}
   end
   if type(x) == "table" and type(x[1]) == "string" then
      -- directory for file operations
      self.loopdir = x[1] .. "/"
   end
end

-- velocity tracking

function arpeggio:update_veltracker(chord, vel)
   if next(chord) == nil then
      -- reset
      self.minavg, self.maxavg = nil, nil
      if self.debug&2~=0 then
	 print(string.format("min = %s, max = %s", self.minavg, self.maxavg))
      end
   elseif vel > 0 then
      -- calculate the velocity envelope
      if not self.minavg then
	 self.minavg = self.minvel
      end
      self.minavg = self.minavg*self.g + vel*(1-self.g)
      if not self.maxavg then
	 self.maxavg = self.maxvel
      end
      self.maxavg = self.maxavg*self.g + vel*(1-self.g)
      if self.debug&2~=0 then
	 print(string.format("vel min = %g, max = %g", self.minavg, self.maxavg))
      end
   end
end

function arpeggio:velrange()
   if self.veltracker ~= 0 then
      local g = self.gain
      local min = self.minavg or self.minvel
      local max = self.maxavg or self.maxvel
      min = g*self.minvel + (1-g)*min
      max = g*self.maxvel + (1-g)*max
      return min, max
   else
      return self.minvel, self.maxvel
   end
end

-- output the next note in the pattern and switch to the next pulse
-- The result is a tuple notes, vel, gate, w, n, where vel is the velocity,
-- gate the gate value (normalized duration), w the pulse weight
-- (indispensability), and n the total number of pulses. The first return
-- value indicates the notes to play. This may either be a singleton number or
-- a list (which can also be empty, or contain multiple note numbers).
function arpeggio:pulse()
   local w, n = self:meter(self.idx)
   -- normalized pulse strength
   local w1 = w/math.max(1,n-1)
   -- corresponding MIDI velocity
   local minvel, maxvel = self:velrange()
   local vel =
      math.floor(mod_value(minvel, maxvel, self.velmod, w1))
   -- The default gate value in the Ardour plugin is always zero which forces
   -- legato mode. This causes notes to hang around indefinitely in some
   -- situations. We use the set (unmodulated) gate parameter value instead,
   -- so that a forced legato will still work if notes are filtered
   -- out. (Maybe this needs reworking in the Ardour plugin as well?)
   local gate, notes = self.gate, nil
   if self.loopstate == 1 and self.loopsize > 0 then
      -- notes come straight from the loop, input is ignored
      notes, vel, gate = table.unpack(self:loop_get())
      self.idx = (self.idx + 1) % self.beats
      return notes, vel, gate, w, n
   end
   if type(self.pattern) == "function" then
      notes = self.pattern(w1)
   elseif next(self.pattern) ~= nil then
      notes = cycle(self.pattern, self.idx+1)
   end
   if notes ~= nil then
      -- note filtering
      local ok = true
      local wmin, wmax = self.wmin, self.wmax
      if w1 >= wmin and w1 <= wmax then
	 local pmin, pmax = self.pmin, self.pmax
	 -- Calculate the filter probablity. We allow for negative pmod values
	 -- here, in which case stronger pulses tend to be filtered out first
	 -- rather than weaker ones.
	 local p = mod_value(pmin, pmax, self.pmod, w1)
	 local r = math.random()
	 if self.debug&4~=0 then
	    print(string.format("w = %g, wmin = %g, wmax = %g, p = %g, r = %g",
				w1, wmin, wmax, p, r))
	 end
	 ok = r <= p
      else
	 ok = false
      end
      if ok then
	 -- modulated gate value
	 gate = mod_value(0, self.gate, self.gatemod, w1)
	 -- output notes (there may be more than one in Raptor mode)
	 if self.debug&4~=0 then
	    print(string.format("idx = %g, notes = %s, vel = %g, gate = %g", self.idx, inspect(notes), vel, gate))
	 end
      else
	 notes = {}
      end
   else
      notes = {}
   end
   self:loop_add(notes, vel, gate)
   self.idx = (self.idx + 1) % self.beats
   return notes, vel, gate, w, n
end

-- panic clears the chord memory and pattern
function arpeggio:panic()
   self.chord = {}
   self.pattern = {}
   self.last_q = nil
   self:set_latch(self.latch and 1 or 0)
   self:update_veltracker({}, 0)
end

-- change the current pulse index
function arpeggio:set_idx(x)
   x = self:intarg(x)
   if type(x) == "number" and self.idx ~= x then
      self.idx = math.max(0, x) % self.beats
      if self.loopstate == 1 then
	 self.loopidx = self.idx % math.max(1, math.min(#self.loop, self.loopsize))
      end
   end
end

-- pattern computation

local function transp(chord, i)
   return map(chord, function (n) return n+12*i end)
end

function arpeggio:pitchrange(a, b)
   if self.pitchtracker == 0 then
      -- just octave range
      a = math.max(0, math.min(127, a+12*self.down))
      b = math.max(0, math.min(127, b+12*self.up))
   elseif self.pitchtracker == 1 then
      -- full range tracker
      a = math.max(0, math.min(127, a+12*self.down+self.pitchlo))
      b = math.max(0, math.min(127, b+12*self.up+self.pitchhi))
   elseif self.pitchtracker == 2 then
      -- treble tracker
      a = math.max(0, math.min(127, b+12*self.down+self.pitchlo))
      b = math.max(0, math.min(127, b+12*self.up+self.pitchhi))
   elseif self.pitchtracker == 3 then
      -- bass tracker
      a = math.max(0, math.min(127, a+12*self.down+self.pitchlo))
      b = math.max(0, math.min(127, a+12*self.up+self.pitchhi))
   end
   return seq(a, b)
end

function arpeggio:create_pattern(chord)
   -- create a new pattern using the current settings
   local pattern = chord
   -- By default we do outside-in by alternating up-down (i.e., lo-hi), set
   -- this flag to true to get something more Logic-like which goes down-up.
   local logic_like = false
   if next(pattern) == nil then
      -- nothing to see here, move along...
      return pattern
   elseif self.raptor ~= 0 then
      -- Raptor mode: Pick random notes from the eligible range based on
      -- average Barlow harmonicities (cf. barlow.lua). This also combines
      -- with mode 0..5, employing the corresponding Raptor arpeggiation
      -- modes. Note that these patterns may contain notes that we're not
      -- actually playing, if they're harmonically related to the input
      -- chord. Raptor can also play chords rather than just single notes, and
      -- with the right settings you can make it go from plain tonal to more
      -- jazz-like and free to completely atonal, and everything in between.
      local a, b = pattern[1], pattern[#pattern]
      -- NOTE: As this kind of pattern is quite costly to compute, we
      -- implement it as a closure which gets evaluated lazily for each pulse,
      -- rather than precomputing the entire pattern at once as in the
      -- deterministic modes.
      if self.mode == 5 then
	 -- Raptor by itself doesn't support mode 5 (outside-in), so we
	 -- emulate it by alternating between mode 1 and 2. This isn't quite
	 -- the same, but it's as close to outside-in as I can make it. You
	 -- might also consider mode 0 (random) as a reasonable alternative
	 -- instead.
	 local cache, mode, dir
	 local function restart()
	    -- print("raptor: restart")
	    cache = {{}, {}}
	    if logic_like then
	       mode, dir = 2, -1
	    else
	       mode, dir = 1, 1
	    end
	 end
	 restart()
	 pattern = function(w1)
	    local notes, _
	    if w1 == 1 then
	       -- beginning of bar, restart pattern
	       restart()
	    end
	    notes, _ =
	       rand_notes(w1,
			  self.nmax, self.nmod,
			  self.hmin, self.hmax, self.hmod,
			  self.smin, self.smax, self.smod,
			  dir, mode, self.uniq ~= 0,
			  self.pref, self.prefmod,
			  cache[mode],
			  chord, self:pitchrange(a, b))
	    if next(notes) ~= nil then
	       cache[mode] = notes
	    end
	    if dir>0 then
	       mode, dir = 2, -1
	    else
	       mode, dir = 1, 1
	    end
	    return notes
	 end
      else
	 local cache, mode, dir
	 local function restart()
	    -- print("raptor: restart")
	    cache = {}
	    mode = self.mode
	    dir = 0
	    if mode == 1 or mode == 3 then
	       dir = 1
	    elseif mode == 2 or mode == 4 then
	       dir = -1
	    end
	 end
	 restart()
	 pattern = function(w1)
	    local notes
	    if w1 == 1 then
	       -- beginning of bar, restart pattern
	       restart()
	    end
	    notes, dir =
	       rand_notes(w1,
			  self.nmax, self.nmod,
			  self.hmin, self.hmax, self.hmod,
			  self.smin, self.smax, self.smod,
			  dir, mode, self.uniq ~= 0,
			  self.pref, self.prefmod,
			  cache,
			  chord, self:pitchrange(a, b))
	    if next(notes) ~= nil then
	       cache = notes
	    end
	    return notes
	 end
      end
   else
      -- apply the octave range (not used in raptor mode)
      pattern = {}
      for i = self.down, self.up do
	 pattern = tabcat(pattern, transp(chord, i))
      end
      if self.mode == 0 then
	 -- random: this is just the run-of-the-mill random pattern permutation
	 local n, pat = #pattern, {}
	 local p = seq(1, n)
	 for i = 1, n do
	    local j = math.random(i, n)
	    p[i], p[j] = p[j], p[i]
	 end
	 for i = 1, n do
	    pat[i] = pattern[p[i]]
	 end
	 pattern = pat
      elseif self.mode == 1 then
	 -- up (no-op)
      elseif self.mode == 2 then
	 -- down
	 pattern = reverse(pattern)
      elseif self.mode == 3 then
	 -- up-down
	 local r = reverse(pattern)
	 -- get rid of the repeated note in the middle
	 table.remove(pattern)
	 pattern = tabcat(pattern, r)
      elseif self.mode == 4 then
	 -- down-up
	 local r = reverse(pattern)
	 table.remove(r)
	 pattern = tabcat(reverse(pattern), pattern)
      elseif self.mode == 5 then
	 -- outside-in
	 local n, pat = #pattern, {}
	 local p, q = n//2, n%2
	 if logic_like then
	    for i = 1, p do
	       -- highest note first (a la Logic?)
	       pat[2*i-1] = pattern[n+1-i]
	       pat[2*i] = pattern[i]
	    end
	 else
	    for i = 1, p do
	       -- lowest note first (sounds better IMHO)
	       pat[2*i-1] = pattern[i]
	       pat[2*i] = pattern[n+1-i]
	    end
	 end
	 if q > 0 then
	    pat[n] = pattern[p+1]
	 end
	 pattern = pat
      end
   end
   if self.debug&1~=0 then
      print(string.format("chord = %s", inspect(chord)))
      print(string.format("pattern = %s", inspect(pattern)))
   end
   return pattern
end

-- latch: keep chord notes when released until new chord or reset
function arpeggio:set_latch(x)
   x = self:intarg(x)
   if type(x) == "number" then
      if x ~= 0 then
	 self.latch = {table.unpack(self.chord)}
      elseif self.latch then
	 self.latch = nil
	 self.pattern = self:create_pattern(self.chord)
      end
   end
end

function arpeggio:get_chord()
   return self.latch and self.latch or self.chord
end

-- change the range of the pattern
function arpeggio:set_up(x)
   x = self:intarg(x)
   if type(x) == "number" then
      self.up = math.max(-2, math.min(2, x))
      self.pattern = self:create_pattern(self:get_chord())
   end
end

function arpeggio:set_down(x)
   x = self:intarg(x)
   if type(x) == "number" then
      self.down = math.max(-2, math.min(2, x))
      self.pattern = self:create_pattern(self:get_chord())
   end
end

function arpeggio:set_pitchtracker(x)
   x = self:intarg(x)
   if type(x) == "number" then
      self.pitchtracker = math.max(0, math.min(3, x))
      self.pattern = self:create_pattern(self:get_chord())
   end
end

function arpeggio:set_pitchlo(x)
   x = self:intarg(x)
   if type(x) == "number" then
      self.pitchlo = math.max(-36, math.min(36, x))
      self.pattern = self:create_pattern(self:get_chord())
   end
end

function arpeggio:set_pitchhi(x)
   x = self:intarg(x)
   if type(x) == "number" then
      self.pitchhi = math.max(-36, math.min(36, x))
      self.pattern = self:create_pattern(self:get_chord())
   end
end

-- change the mode (up, down, etc.)
function arpeggio:set_mode(x)
   x = self:intarg(x)
   if type(x) == "number" then
      self.mode = math.max(0, math.min(5, x))
      self.pattern = self:create_pattern(self:get_chord())
   end
end

-- this enables Raptor mode with randomized note output
function arpeggio:set_raptor(x)
   x = self:intarg(x)
   if type(x) == "number" then
      self.raptor = math.max(0, math.min(1, x))
      self.pattern = self:create_pattern(self:get_chord())
   end
end

-- change min/max velocities, gate, and note probabilities
function arpeggio:set_minvel(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.minvel = math.max(0, math.min(127, x))
   end
end

function arpeggio:set_maxvel(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.maxvel = math.max(0, math.min(127, x))
   end
end

function arpeggio:set_velmod(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.velmod = math.max(-1, math.min(1, x))
   end
end

function arpeggio:set_veltracker(x)
   x = self:intarg(x)
   if type(x) == "number" then
      self.veltracker = math.max(0, math.min(1, x))
   end
end

function arpeggio:set_gain(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.gain = math.max(0, math.min(1, x))
   end
end

function arpeggio:set_gate(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.gate = math.max(0, math.min(10, x))
   end
end

function arpeggio:set_gatemod(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.gatemod = math.max(-1, math.min(1, x))
   end
end

function arpeggio:set_pmin(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.pmin = math.max(0, math.min(1, x))
   end
end

function arpeggio:set_pmax(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.pmax = math.max(0, math.min(1, x))
   end
end

function arpeggio:set_pmod(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.pmod = math.max(-1, math.min(1, x))
   end
end

function arpeggio:set_wmin(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.wmin = math.max(0, math.min(1, x))
   end
end

function arpeggio:set_wmax(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.wmax = math.max(0, math.min(1, x))
   end
end

-- change the raptor parameters (harmonicity, etc.)
function arpeggio:set_nmax(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.nmax = math.max(0, math.min(10, x))
   end
end

function arpeggio:set_nmod(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.nmod = math.max(-1, math.min(1, x))
   end
end

function arpeggio:set_hmin(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.hmin = math.max(0, math.min(1, x))
   end
end

function arpeggio:set_hmax(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.hmax = math.max(0, math.min(1, x))
   end
end

function arpeggio:set_hmod(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.hmod = math.max(-1, math.min(1, x))
   end
end

function arpeggio:set_smin(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.smin = math.max(-127, math.min(127, x))
   end
end

function arpeggio:set_smax(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.smax = math.max(-127, math.min(127, x))
   end
end

function arpeggio:set_smod(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.smod = math.max(-1, math.min(1, x))
   end
end

function arpeggio:set_uniq(x)
   x = self:intarg(x)
   if type(x) == "number" then
      self.uniq = math.max(0, math.min(1, x))
   end
end

function arpeggio:set_pref(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.pref = math.max(-1, math.min(1, x))
   end
end

function arpeggio:set_prefmod(x)
   x = self:numarg(x)
   if type(x) == "number" then
      self.prefmod = math.max(-1, math.min(1, x))
   end
end

local function update_chord(chord, note, vel)
   -- update the chord memory, keeping the notes in ascending order
   local n = #chord
   if n == 0 then
      if vel > 0 then
	 table.insert(chord, 1, note)
      end
      return chord
   end
   for i = 1, n do
      if chord[i] == note then
	 if vel <= 0 then
	    -- note off: remove note
	    if i < n then
	       table.move(chord, i+1, n, i)
	    end
	    table.remove(chord)
	 end
	 return chord
      elseif chord[i] > note then
	 if vel > 0 then
	    -- insert note
	    table.insert(chord, i, note)
	 end
	 return chord
      end
   end
   -- if we come here, no note has been inserted or deleted yet
   if vel > 0 then
      -- note is larger than all present notes in chord, so it needs to be
      -- inserted at the end
      table.insert(chord, note)
   end
   return chord
end

-- note input; update the internal chord memory and recompute the pattern
function arpeggio:note(note, vel)
   if self.debug&2~=0 then
      print(string.format("note = %s", inspect({ note, vel })))
   end
   if type(note) == "number" and type(vel) == "number" then
      if self.latch and next(self.chord) == nil and vel>0 then
	 -- start new pattern
	 self.latch = {}
      end
      update_chord(self.chord, note, vel)
      if self.latch and vel>0 then
	 update_chord(self.latch, note, vel)
      end
      self.pattern = self:create_pattern(self:get_chord())
      self:update_veltracker(self:get_chord(), vel)
   end
end

-- this recomputes all indispensability tables
function arpeggio:prepare_meter(meter)
   local n = 1
   local m = {}
   if type(meter) ~= "table" then
      -- assume singleton number
      meter = { meter }
   end
   for _,q in ipairs(meter) do
      if q ~= math.floor(q) then
	 error("arpeggio: meter levels must be integer")
	 return
      elseif q < 1 then
	 error("arpeggio: meter levels must be positive")
	 return
      end
      -- factorize each level as Barlow's formula assumes primes
      m = tabcat(m, factor(q))
      n = n*q
   end
   self.beats = n
   self.last_q = nil
   if n > 1 then
      self.indisp[1] = indisp(m)
      for q = 2, self.n do
	 local qs = tabcat(m, factor(q))
	 self.indisp[q] = indisp(qs)
      end
   else
      self.indisp[1] = {0}
      for q = 2, self.n do
	 self.indisp[q] = indisp(q)
      end
   end
end

-- set a new meter (given either as a singleton number or as a list of
-- numbers) and return the number of pulses
function arpeggio:set_meter(meter)
   self:prepare_meter(meter)
   return self.beats
end

-- -------------------------------------------------------------------------

-- Pd interface

local pdx = require 'pdx'

-- debug level: This only affects the plugin code. The available levels are:
-- 1: print preset changes only, 2: also print the current beat and other
-- important state information, 3: also print note output, 4: print
-- everything, including note input. Output goes to the Pd console.
-- NOTE: To debug the internal state of the arpeggiator object, including
-- pattern changes and note generation, use the arp.debug setting below.
local debug = 1

-- Parameter and preset tables. These are the same as in the Ardour plugin.
-- Note that some of the fields aren't used in the Pd implementation.

local hrm_scalepoints = { ["0.09 (minor 7th and 3rd)"] = 0.09, ["0.1 (major 2nd and 3rd)"] = 0.1, ["0.17 (4th)"] = 0.17, ["0.21 (5th)"] = 0.21, ["1 (unison, octave)"] = 1 }

local params = {
   { type = "input", name = "bypass", min = 0, max = 1, default = 0, toggled = true, doc = "bypass the arpeggiator, pass through input notes" },
   { type = "input", name = "division", min = 1, max = 7, default = 1, integer = true, doc = "number of subdivisions of the beat" },
   { type = "input", name = "pgm", min = 0, max = 128, default = 0, integer = true, doc = "program change", scalepoints = { default = 0 } },
   { type = "input", name = "latch", min = 0, max = 1, default = 0, toggled = true, doc = "toggle latch mode" },
   { type = "input", name = "up", min = -2, max = 2, default = 1, integer = true, doc = "octave range up" },
   { type = "input", name = "down", min = -2, max = 2, default = -1, integer = true, doc = "octave range down" },
   -- This isn't in the Ardour plugin, but it's occasionally useful to have
   -- the option to transpose notes a given number of semitones up or down in
   -- the stand-alone version, so that's what this option is for.
   { type = "input", name = "transp", min = -64, max = 64, default = 0, integer = true, doc = "transpose by given number of semitones" },
   -- Raptor's usual default for the pattern is 0 = random, but 1 = up
   -- seems to be a more sensible choice.
   { type = "input", name = "mode", min = 0, max = 5, default = 1, enum = true, doc = "pattern style",
     scalepoints =
	{ ["0 random"] = 0, ["1 up"] = 1, ["2 down"] = 2, ["3 up-down"] = 3, ["4 down-up"] = 4, ["5 outside-in"] = 5 } },
   { type = "input", name = "raptor", min = 0, max = 1, default = 0, toggled = true, doc = "toggle raptor mode" },
   { type = "input", name = "minvel", min = 0, max = 127, default = 60, integer = true, doc = "minimum velocity" },
   { type = "input", name = "maxvel", min = 0, max = 127, default = 120, integer = true, doc = "maximum velocity" },
   { type = "input", name = "velmod", min = -1, max = 1, default = 1, doc = "automatic velocity modulation according to current pulse strength" },
   { type = "input", name = "gain", min = 0, max = 1, default = 1, doc = "wet/dry mix between input velocity and set values (min/max velocity)" },
   -- The original Pd Raptor allows this to go from 0 to 1000%, but we only
   -- support 0-100% here.
   { type = "input", name = "gate", min = 0, max = 1, default = 1, doc = "gate as fraction of pulse length", scalepoints = { legato = 0 } },
   { type = "input", name = "gatemod", min = -1, max = 1, default = 0, doc = "automatic gate modulation according to current pulse strength" },
   { type = "input", name = "wmin", min = 0, max = 1, default = 0, doc = "minimum note weight" },
   { type = "input", name = "wmax", min = 0, max = 1, default = 1, doc = "maximum note weight" },
   { type = "input", name = "pmin", min = 0, max = 1, default = 0.3, doc = "minimum note probability" },
   { type = "input", name = "pmax", min = 0, max = 1, default = 1, doc = "maximum note probability" },
   { type = "input", name = "pmod", min = -1, max = 1, default = 0, doc = "automatic note probability modulation according to current pulse strength" },
   { type = "input", name = "hmin", min = 0, max = 1, default = 0, doc = "minimum harmonicity", scalepoints = hrm_scalepoints },
   { type = "input", name = "hmax", min = 0, max = 1, default = 1, doc = "maximum harmonicity", scalepoints = hrm_scalepoints },
   { type = "input", name = "hmod", min = -1, max = 1, default = 0, doc = "automatic harmonicity modulation according to current pulse strength" },
   { type = "input", name = "pref", min = -1, max = 1, default = 1, doc = "harmonic preference" },
   { type = "input", name = "prefmod", min = -1, max = 1, default = 0, doc = "automatic harmonic preference modulation according to current pulse strength" },
   { type = "input", name = "smin", min = -12, max = 12, default = 1, integer = true, doc = "minimum step size" },
   { type = "input", name = "smax", min = -12, max = 12, default = 7, integer = true, doc = "maximum step size" },
   { type = "input", name = "smod", min = -1, max = 1, default = 0, doc = "automatic step size modulation according to current pulse strength" },
   { type = "input", name = "nmax", min = 0, max = 10, default = 1, integer = true, doc = "maximum polyphony (number of simultaneous notes)" },
   { type = "input", name = "nmod", min = -1, max = 1, default = 0, doc = "automatic modulation of the number of notes according to current pulse strength" },
   { type = "input", name = "uniq", min = 0, max = 1, default = 1, toggled = true, doc = "don't repeat notes in consecutive steps" },
   { type = "input", name = "pitchhi", min = -36, max = 36, default = 0, integer = true, doc = "extended pitch range up in semitones (raptor mode)" },
   { type = "input", name = "pitchlo", min = -36, max = 36, default = 0, integer = true, doc = "extended pitch range down in semitones (raptor mode)" },
   { type = "input", name = "pitchtracker", min = 0, max = 3, default = 0, enum = true, doc = "pitch tracker mode, follow input to adjust the pitch range (raptor mode)",
     scalepoints =
	{ ["0 off"] = 0, ["1 on"] = 1, ["2 treble"] = 2, ["3 bass"] = 3 } },
   { type = "input", name = "inchan", min = 0, max = 128, default = 0, integer = true, doc = "input channel (0 = omni = all channels)", scalepoints = { omni = 0 } },
   { type = "input", name = "outchan", min = 0, max = 128, default = 0, integer = true, doc = "input channel (0 = omni = input channel)", scalepoints = { omni = 0 } },
   { type = "input", name = "loopsize", min = 0, max = 16, default = 4, integer = true, doc = "loop size (number of bars)" },
   { type = "input", name = "loop", min = 0, max = 1, default = 0, toggled = true, doc = "toggle loop mode" },
   { type = "input", name = "mute", min = 0, max = 1, default = 0, toggled = true, doc = "turn the arpeggiator off, suppress all note output" },
}

local n_params = #params
local int_param = map(params, function(x) return x.integer == true or x.enum == true or x.toggled == true end)

-- This is basically a collection of presets from the original Pd external,
-- with some (very) minor adjustments / bugfixes where I saw fit. The program
-- numbers assume a GM patch set, if your synth isn't GM-compatible then
-- you'll have to adjust them accordingly. NOTE: The tr808 preset assumes a
-- GM-compatible drumkit, so it outputs through MIDI channel 10 by default;
-- other presets leave the output channel as is.

local raptor_presets = {
   { name = "default", params = { bypass = 0, latch = 0, division = 1, pgm = 0, up = 1, down = -1, transp = 0, mode = 1, raptor = 0, minvel = 60, maxvel = 120, velmod = 1, gain = 1, gate = 1, gatemod = 0, wmin = 0, wmax = 1, pmin = 0.3, pmax = 1, pmod = 0, hmin = 0, hmax = 1, hmod = 0, pref = 1, prefmod = 0, smin = 1, smax = 7, smod = 0, nmax = 1, nmod = 0, uniq = 1, pitchhi = 0, pitchlo = 0, pitchtracker = 0, inchan = 0, outchan = 0, loopsize = 4, loop = 0, mute = 0 } },
   { name = "arp", params = { pgm = 26, up = 0, down = -1, mode = 3, raptor = 1, minvel = 105, maxvel = 120, velmod = 1, gain = 0.5, gate = 1, gatemod = 0, wmin = 0, wmax = 1, pmin = 0.9, pmax = 1, pmod = -1, hmin = 0.11, hmax = 1, hmod = 0, pref = 0.8, prefmod = 0, smin = 2, smax = 7, smod = 0, nmax = 1, nmod = 0, uniq = 1, pitchhi = 0, pitchlo = -12, pitchtracker = 2, loopsize = 4 } },
   { name = "bass", params = { pgm = 35, up = 0, down = -1, mode = 3, raptor = 1, minvel = 40, maxvel = 120, velmod = 1, gain = 0.5, gate = 1, gatemod = 0, wmin = 0, wmax = 1, pmin = 0.2, pmax = 1, pmod = 1, hmin = 0.12, hmax = 1, hmod = 0.1, pref = 0.8, prefmod = 0.1, smin = 2, smax = 7, smod = 0, nmax = 1, nmod = 0, uniq = 1, pitchhi = 7, pitchlo = 0, pitchtracker = 3, loopsize = 4 } },
   { name = "piano", params = { pgm = 1, up = 1, down = -1, mode = 0, raptor = 1, minvel = 90, maxvel = 120, velmod = 1, gain = 0.5, gate = 1, gatemod = 0, wmin = 0, wmax = 1, pmin = 0.4, pmax = 1, pmod = 1, hmin = 0.14, hmax = 1, hmod = 0.1, pref = 0.6, prefmod = 0.1, smin = 2, smax = 5, smod = 0, nmax = 2, nmod = 0, uniq = 1, pitchhi = 0, pitchlo = -18, pitchtracker = 2, loopsize = 4 } },
   { name = "raptor", params = { pgm = 5, up = 1, down = -2, mode = 0, raptor = 1, minvel = 60, maxvel = 120, velmod = 1, gain = 0.5, gate = 1, gatemod = 0, wmin = 0, wmax = 1, pmin = 0.4, pmax = 0.9, pmod = 0, hmin = 0.09, hmax = 1, hmod = -1, pref = 1, prefmod = 1, smin = 1, smax = 7, smod = 0, nmax = 3, nmod = -1, uniq = 0, pitchhi = 0, pitchlo = 0, pitchtracker = 0, loopsize = 4 } },
   -- some variations of the raptor preset for different instruments
   { name = "raptor-arp", params = { pgm = 26, up = 0, down = -1, mode = 3, raptor = 1, minvel = 105, maxvel = 120, velmod = 1, gain = 0.5, gate = 1, gatemod = 0, wmin = 0, wmax = 1, pmin = 0.4, pmax = 0.9, pmod = 0, hmin = 0.09, hmax = 1, hmod = -1, pref = 1, prefmod = 1, smin = 2, smax = 7, smod = 0, nmax = 1, nmod = 0, uniq = 1, pitchhi = 0, pitchlo = -12, pitchtracker = 2, loopsize = 4 } },
   { name = "raptor-bass", params = { pgm = 35, up = 0, down = -1, mode = 3, raptor = 1, minvel = 40, maxvel = 120, velmod = 1, gain = 0.5, gate = 1, gatemod = 0, wmin = 0, wmax = 1, pmin = 0.4, pmax = 0.9, pmod = 0, hmin = 0.09, hmax = 1, hmod = -1, pref = 1, prefmod = -0.6, smin = 2, smax = 7, smod = 0, nmax = 1, nmod = 0, uniq = 1, pitchhi = 7, pitchlo = -6, pitchtracker = 3, loopsize = 4 } },
   { name = "raptor-piano", params = { pgm = 1, up = 1, down = -1, mode = 0, raptor = 1, minvel = 90, maxvel = 120, velmod = 1, gain = 0.5, gate = 1, gatemod = 0, wmin = 0, wmax = 1, pmin = 0.4, pmax = 0.9, pmod = 0, hmin = 0.09, hmax = 1, hmod = -1, pref = -0.4, prefmod = -0.6, smin = 2, smax = 5, smod = 0, nmax = 2, nmod = 0, uniq = 1, pitchhi = 0, pitchlo = -18, pitchtracker = 2, loopsize = 4 } },
   { name = "raptor-solo", params = { pgm = 25, up = 0, down = -1, mode = 3, raptor = 1, minvel = 40, maxvel = 110, velmod = 0.5, gain = 0.5, gate = 1, gatemod = 0.5, wmin = 0, wmax = 1, pmin = 0.2, pmax = 0.9, pmod = 0.5, hmin = 0.09, hmax = 1, hmod = -1, pref = -0.4, prefmod = 0, smin = 1, smax = 7, smod = 0, nmax = 1, nmod = 0, uniq = 1, pitchhi = 0, pitchlo = 0, pitchtracker = 0, loopsize = 4 } },
   { name = "tr808", params = { pgm = 26, outchan = 10, up = 0, down = 0, mode = 1, raptor = 0, minvel = 60, maxvel = 120, velmod = 1, gain = 0.5, gate = 1, gatemod = 0, wmin = 0, wmax = 1, pmin = 0.3, pmax = 1, pmod = 0, hmin = 0, hmax = 1, hmod = 0, pref = 1, prefmod = 0, smin = 1, smax = 7, smod = 0, nmax = 1, nmod = 0, uniq = 1, pitchhi = 0, pitchlo = 0, pitchtracker = 0, loopsize = 4 } },
   { name = "vibes", params = { pgm = 12, up = 0, down = -1, mode = 3, raptor = 1, minvel = 84, maxvel = 120, velmod = 1, gain = 0.5, gate = 1, gatemod = 0, wmin = 0, wmax = 1, pmin = 0.9, pmax = 1, pmod = -1, hmin = 0.14, hmax = 1, hmod = 0.1, pref = 0.6, prefmod = 0.1, smin = 2, smax = 5, smod = 0, nmax = 2, nmod = 0, uniq = 1, pitchhi = -5, pitchlo = -16, pitchtracker = 2, loopsize = 4 } },
   { name = "weirdmod", params = { pgm = 25, up = 0, down = -1, mode = 5, raptor = 0, minvel = 40, maxvel = 110, velmod = 0.5, gain = 0.5, gate = 1, gatemod = 0.5, wmin = 0, wmax = 1, pmin = 0.2, pmax = 0.9, pmod = 0.5, hmin = 0, hmax = 1, hmod = 0, pref = 1, prefmod = 0, smin = 1, smax = 7, smod = 0, nmax = 1, nmod = 0, uniq = 1, pitchhi = 0, pitchlo = 0, pitchtracker = 0, loopsize = 4 } },
}

local n_presets = #raptor_presets

-- param and preset indices
local param_i = {}
for i = 1, n_params do
   param_i[params[i].name] = i
end

local preset_i = {}
for i = 1, n_presets do
   preset_i[raptor_presets[i].name] = i
end

-- table of params to be skipped when loading and saving presets

-- Basically, these are performance controls (various status toggles such as
-- bypass and mute) which are to be operated during live performance. Thus we
-- don't want to load these with the preset in order to not disrupt the live
-- performance. This is also the case for meter and subdivision -- if you need
-- to change these around quickly, you can do that by sending a meter message
-- to the raptor object instead.
local param_skip = {}
-- status toggles
param_skip["bypass"] = true
param_skip["mute"] = true
param_skip["latch"] = true
param_skip["loop"] = true
-- division parameter (no need to exclude meter here, since it isn't in the
-- arpeggiator params)
param_skip["division"] = true

-- param setters

local function arp_set_loopsize(self, x)
   -- need to translate beat numbers to steps
   self:set_loopsize(x*self.beats)
end

function raptor:set(param, x)
   -- this sets various parameters which actually live in the raptor instance,
   -- not the arpeggiator
   local last_bypass = self.bypass
   local last_mute = self.mute
   local last_division = self.division
   local last_inchan = self.inchan
   local last_pgm = self.pgm
   self[param] = x
   -- various state changes that need special treatment
   if (self.bypass ~= last_bypass and self.bypass ~= 0) or
      (self.mute ~= last_mute and self.mute ~= 0) then
      -- turn off any sounding notes from the arpeggiator
      self:notes_off()
   end
   if self.division ~= last_division then
      -- update the meter
      self:update_meter()
   end
   if self.inchan ~= last_inchan and self.inchan > 0 then
      -- change of input channel, kill off chord memory and stop notes
      self.arp:panic()
      self:notes_off()
   end
   if self.pgm ~= last_pgm or self:get_chan(self.chan) ~= self.chan then
      -- program or output channel has changed, send the program change
      self.chan = self:get_chan(self.chan)
      if self.pgm > 0 then
	 self:outlet(1, "pgm", { self.pgm, self.chan })
      end
   end
end

function raptor:set_param_tables()
   -- this initializes the parameter setter callbacks; this needs to be redone
   -- after reloading the object (pdx.reload)
   self.param_set = { self.set, self.set, self.set, self.arp.set_latch, self.arp.set_up, self.arp.set_down, self.set, self.arp.set_mode, self.arp.set_raptor, self.arp.set_minvel, self.arp.set_maxvel, self.arp.set_velmod, self.arp.set_gain, self.arp.set_gate, self.arp.set_gatemod, self.arp.set_wmin, self.arp.set_wmax, self.arp.set_pmin, self.arp.set_pmax, self.arp.set_pmod, self.arp.set_hmin, self.arp.set_hmax, self.arp.set_hmod, self.arp.set_pref, self.arp.set_prefmod, self.arp.set_smin, self.arp.set_smax, self.arp.set_smod, self.arp.set_nmax, self.arp.set_nmod, self.arp.set_uniq, self.arp.set_pitchhi, self.arp.set_pitchlo, self.arp.set_pitchtracker, self.set, self.set, arp_set_loopsize, self.arp.set_loop, self.set }
end

function raptor:initialize(sel, atoms)
   pdx.reload(self)

   self.inlets = 1
   self.outlets = 3

   -- initialize param values
   self.param_val = {}
   for i = 1, n_params do
      self.param_val[i] = params[i].default
   end

   -- these are maintained in the Pd object
   self.bypass = 0
   self.mute = 0

   -- default  meter (numerator, denominator, and subdivision) and tempo
   self.n = 4
   self.m = 4
   self.division = 1
   self.tempo = 120

   -- create the arpeggiator (default meter)
   self.arp = arpeggio:new(self.n * self.division)

   -- Debugging output from the arpeggiator object (bitmask):
   -- 1 = pattern, 2 = input, 4 = output (e.g., 7 means "all")
   -- This is intended for debugging purposes only. it spits out *a lot* of
   -- cryptic debug messages in the log window, so it's better to keep this
   -- disabled in production code.
   --self.arp.debug = 7

   -- set up the callback tables
   self:set_param_tables()

   -- last output notes and channel
   self.last_notes = nil
   self.last_chan = nil

   -- midi parameters
   self.pgm = 0
   self.inchan = 0
   self.outchan = 0
   self.chan = 1
   self.transp = 0

   -- midi learn
   self.midi_map = {}
   self.midi_learn = 0
   self.midi_learn_cc = nil
   self.midi_learn_ch = nil
   self.midi_learn_var = nil
   self:load_map()

   -- instance id; this gets initialized later by the dump method, see below
   self.id = nil

   -- initialize the user presets
   self:load_presets()

   -- create a global receiver, so that we can tell all instances about global
   -- state changes
   self.recv = pd.Receive:new():register(self, "__raptor", "receive")

   -- initialize the note-off timer
   self.clock = pd.Clock:new():register(self, "notes_off")

   return true
end

function raptor:finalize()
   self.clock:destruct()
   self.recv:destruct()
end

-- pulses

function raptor:notes_off()
   if self.last_notes then
      -- kill the old notes
      for _, num in ipairs(self.last_notes) do
	 if debug >= 3 then
	    print(string.format("[out] note off %d", num))
	 end
	 self:outlet(1, "note", { num, 0, self.last_chan })
      end
      self.last_notes = nil
      -- stop the note-off timer in case it's still pending
      self.clock:unset()
   end
end

function raptor:in_1_bang()
   -- grab some notes from the arpeggiator
   local p = self.arp.idx
   local notes, vel, gate, w, n = self.arp:pulse()
   -- check if we're bypassed or muted
   if self.bypass ~= 0 or self.mute ~= 0 then
      return
   end
   if debug >= 2 then
      -- print some debugging information: fractional beat number, current
      -- meter, current tempo
      print (string.format("%g - %d/%d - %g bpm",
			   math.floor(p/self.division*1000)/1000,
			   self.n, self.m, self.tempo))
   end
   -- Make sure that the gate is clamped to the 0-1 range, since we don't
   -- support overlapping notes in the current implementation.
   gate = math.max(0, math.min(1, gate))
   -- calculate the note-off time in ms
   local delay = 60000/self.tempo * 4/self.m/self.division
   local gate_time = delay * gate
   --print(string.format("[%d] notes %s %d %g %g %d", p, inspect(notes), vel, gate, w, n))
   -- the arpeggiator may return a singleton note, make sure that it's always
   -- a list
   if type(notes) ~= "table" then
      notes = { notes }
   end
   -- we take a zero gate value to mean legato instead, in which case notes
   -- extend to the next unfiltered note
   local legato = gate == 0
   if not legato then
      self:notes_off()
   end
   if next(notes) ~= nil then
      if legato then
	 self:notes_off()
      end
      -- output the delay time until the next pulse is due on outlet #3
      self:outlet(3, "float", { delay })
      -- output the current pulse number and number of beats on outlet #2
      self:outlet(2, "list", { p, n })
      -- output the notes on outlet #1
      for i = 1, #notes do
	 local num = notes[i]+self.transp -- apply transposition
	 notes[i] = num
	 if debug >= 3 then
	    print(string.format("[out] note on %d %d", num, vel))
	 end
	 self:outlet(1, "note", { num, vel, self.chan })
      end
      self.last_notes = notes
      self.last_chan = self.chan
      if gate < 1 and not legato then
	 -- Set the time at which the note-offs are due.
	 -- Otherwise no timer is set in which case the
	 -- note-offs get triggered automatically above.
	 self.clock:delay(gate_time)
      end
      if debug >= 2 then
	 -- monitor memory usage of the Lua interpreter
	 print(string.format("mem: %0.2f KB", collectgarbage("count")))
      end
   end
end

-- (re)set the pulse index

function raptor:in_1_float(p)
   self.arp:set_idx(p)
end

function raptor:in_1_reset()
   self.arp:set_idx(0)
end

-- panic -- this resets the arpeggiator and stops all sounding notes

function raptor:in_1_panic()
   self.arp:panic()
   self:notes_off()
end

-- stop -- this just stops all sounding notes, but keeps the arpeggiator state

function raptor:in_1_stop()
   self:notes_off()
end

-- reload -- update the internal state of an instance after global state
-- changes (midi map, user presets, pdx.reload)

function raptor:in_1_reload()
   -- reinitialize the callback tables
   self:set_param_tables()
   -- reload the user presets
   self:load_presets()
   -- reload the midi map
   self:load_map()
end

-- global receiver -- at present we use this for global status updates

function raptor:receive(sel, atoms)
   if sel == "reload" then
      self:in_1_reload()
   elseif sel == "presets" then
      self:load_presets()
   elseif sel == "midimap" then
      self:load_map()
   end
end

-- presets

-- We manage both factory and user presets here. The former are in a static
-- global table which always remains the same across different instances (see
-- above). The latter are read from a file during initialization and are
-- maintained as a dynamic member variable separately for each instance.
-- NOTE: There is only one file for the user presets across all instances, but
-- since the contents of that file may change during operation, we reload
-- instances to sync up their user presets when needed.

function raptor:recall_preset(preset)
   local i = preset_i[preset.name]
   if self.user_preset_i[preset.name] then
      i = self.user_preset_i[preset.name] + n_presets
   end
   if debug >= 1 then
      print(string.format("preset #%d: %s", i, preset.name))
   end
   local function check(var, val)
      if param_skip[var] then
	 return false
      elseif (var == "inchan" or var == "outchan") and val == 0 then
	 -- In order to not disrupt live performances, we don't recall these
	 -- if zero (i.e., not an actual MIDI channel). However, in contrast
	 -- to the other performance parameters, the MIDI channels do get
	 -- recorded in presets, and can be changed using the presets if they
	 -- have a proper (nonzero) value.
	 return false
      else
	 return true
      end
   end
   for var, val in pairs(preset.params) do
      if check(var, val) then
	 --print(string.format("%s = %s", var, tostring(val)))
	 self:param(var, val)
	 if self.id then
	    -- send the parameter so that it can be picked up by the panel
	    pd.send(string.format("%s-%s", self.id, var), "set", {val})
	 end
      end
   end
   if self.id then
      pd.send(string.format("%s-%s", self.id, "preset"), "symbol", {preset.name})
      pd.send(string.format("%s-%s", self.id, "presetno"), "set", {i-1})
   end
end

function raptor:get_preset(i)
   if type(i) == "number" and math.floor(i) == i then
      if raptor_presets[i] then
	 return raptor_presets[i]
      elseif self.user_presets[i-n_presets] then
	 return self.user_presets[i-n_presets]
      end
   elseif type(i) == "string" then
      -- first scan the user presets so that these can overide factory presets
      -- with the same name
      if self.user_preset_i[i] then
	 return self.user_presets[self.user_preset_i[i]]
      elseif preset_i[i] then
	 return raptor_presets[preset_i[i]]
      else
	 return nil
      end
   else
      return nil
   end
end

function raptor:in_1_preset(atoms)
   local preset = self:get_preset(atoms[1])
   if preset then
      self:recall_preset(preset)
   else
      -- print the names of the available presets in the console
      print("factory presets:")
      for i, preset in ipairs(raptor_presets) do
	 print(string.format("%d: %s", i, preset.name))
      end
      if #self.user_presets > 0 then
	 print("user presets:")
	 for i, preset in ipairs(self.user_presets) do
	    print(string.format("%d: %s", i+n_presets, preset.name))
	 end
      end
   end
end

-- save user presets

function raptor:in_1_save(atoms)
   local name = atoms[1]
   if type(name) == "string" and string.len(name) > 0 then
      local preset = { name = name, params = {} }
      for i, param in ipairs(params) do
	 if not param_skip[param.name] then
	    preset.params[param.name] = self.param_val[i]
	 end
      end
      table.insert(self.user_presets, preset)
      local i = #self.user_presets
      self.user_preset_i[name] = i
      print(string.format("saved preset #%d: %s", i+n_presets, name))
      -- write the new preset to the preset file
      local fname = self._canvaspath .. "lib/presets"
      local fp = io.open(fname, "a")
      fp:write(inspect(preset), "\n")
      fp:close()
      -- broadcast a message to all raptor instances so that they can update
      -- themselves
      pd.send("__raptor", "presets", {})
   end
end

-- load users presets

function raptor:load_presets()
   -- load the user presets from the preset file if present
   self.user_presets = {}
   self.user_preset_i = {}
   local fname = self._canvaspath .. "lib/presets"
   local fp = io.open(fname, "r")
   if fp then
      local line = fp:read()
      while line do
	 local f = load("return " .. line)
	 if type(f) == "function" then
	    local preset = f()
	    -- do some quick plausability checks
	    if type(preset) == "table" and type(preset.name) == "string" and
	       string.len(preset.name) > 0 and
	       type(preset.params) == "table" then
	       table.insert(self.user_presets, preset)
	       local i = #self.user_presets
	       self.user_preset_i[preset.name] = i
	    end
	 end
	 line = fp:read()
      end
      fp:close()
   end
end

-- note input (SMMF format)

function raptor:get_chan(ch)
   if self.outchan == 0 and self.inchan > 0 then
      ch = self.inchan -- outchan == inchan > 0 override
   elseif self.outchan > 0 then
      ch = self.outchan -- outchan > 0 override
   end
   return ch
end

function raptor:check_chan(ch)
   return self.inchan == 0 or ch == self.inchan
end

function raptor:in_1_note(atoms)
   if self.bypass ~= 0 then
      -- pass through incoming notes (with transposition applied)
      local num, val, ch = table.unpack(atoms)
      self:outlet(1, "note", {num+self.transp, val, ch})
   else
      local num, val, ch = table.unpack(atoms)
      if not ch then
	 -- default channel
	 ch = 1
      end
      if type(num) == "number" and type(val) == "number" and type(ch) == "number" and self:check_chan(ch) then
	 if debug >= 4 then
	    if val > 0 then
	       print(string.format("[in] note on %d %d", num, val))
	    else
	       print(string.format("[in] note off %d", num))
	    end
	 end
	 self.arp:note(num, val)
	 self.chan = self:get_chan(ch)
      end
   end
end

-- pass through other incoming MIDI messages (CC, pitch bend, etc.)

function raptor:in_1_ctl(atoms)
   if self.midi_learn == 1 then
      -- midi learn for CC
      if atoms[1] > 0 and
	 (not self.midi_learn_cc or
	  self.midi_learn_cc ~= atoms[2] or
	  self.midi_learn_ch ~= atoms[3]) then
	 self.midi_learn_cc = atoms[2]
	 self.midi_learn_ch = atoms[3]
	 self:learn()
	 return
      end
   end
   local var = self:map_get(atoms[2], atoms[3])
   if var then
      -- apply existing mapping
      local val = atoms[1]
      local i = param_i[var]
      if i then
	 if params[i].toggled then
	    if val > 0 then
	       self:param(var, self.param_val[i] == 0 and 1 or 0)
	    end
	 else
	    -- make sure that 64 gets mapped to the half-way value
	    local min, max = params[i].min, params[i].max
	    val = val==127 and max or val/128*(max-min)+min
	    if params[i].integer then
	       val = math.floor(val+0.5)
	    end
	    self:param(var, val)
	 end
	 return
      end
   end
   -- simple pass-through
   self:outlet(1, "ctl", atoms)
end

function raptor:in_1_pgm(atoms)
   -- kludge: this can be either an SMMF or a parameter set/get message, we
   -- deal with that here on the fly
   if #atoms > 1 then
      self:outlet(1, "pgm", atoms)
   else
      self:in_1("pgm", atoms)
   end
end

function raptor:in_1_bend(atoms)
   self:outlet(1, "bend", atoms)
end

function raptor:in_1_touch(atoms)
   self:outlet(1, "touch", atoms)
end

function raptor:in_1_polytouch(atoms)
   local num, val, ch = table.unpack(atoms)
   self:outlet(1, "polytouch", {num+self.transp, val, ch})
end

function raptor:in_1_sysex(atoms)
   self:outlet(1, "sysex", atoms)
end

-- instance parameters (these need special treatment)

function raptor:in_1_tempo(atoms)
   if #atoms == 0 then
      -- report the current value
      self:outlet(1, "float", {self.tempo})
   elseif type(atoms[1]) == "number" then
      -- make sure that this is positive
      self.tempo = math.max(1, atoms[1])
      if self.id then
	 -- make sure to update the panel as well
	 local id = self.id
	 pd.send(string.format("%s-%s", id, "tempo"), "set", {self.tempo})
      end
   end
end

function raptor:update_meter()
   -- update the meter in the arpeggiator
   self.arp:set_meter(self.n*self.division)
   -- we also need to update the loop size here
   local loopsize = self.param_val[n_params-2]
   arp_set_loopsize(self.arp, loopsize)
end

function raptor:in_1_meter(atoms)
   if #atoms == 0 then
      -- report the current value
      self:outlet(1, "list", {self.n, self.m, self.division})
   else
      local n, m, division = self.n, self.m, self.division
      if type(atoms[1]) == "number" then
	 -- make sure that this is a positive integer
	 n = math.max(1, math.floor(atoms[1]))
	 -- second optional argument is the denominator
	 if type(atoms[2]) == "number" then
	    m = math.max(1, math.floor(atoms[2]))
	    -- third optional argument is the subdivison
	    if type(atoms[3]) == "number" then
	       division = math.max(1, math.floor(atoms[3]))
	    end
	 end
      end
      local check = n*division ~= self.n*self.division
      self.n = n
      self.m = m
      self.division = division
      if check then
	 -- update the meter
	 self:update_meter()
      end
      if self.id then
	 -- make sure to update the panel as well
	 local id = self.id
	 pd.send(string.format("%s-%s", id, "meter-num"), "set", {self.n})
	 pd.send(string.format("%s-%s", id, "meter-denom"), "set", {self.m})
	 pd.send(string.format("%s-%s", id, "division"), "set", {self.division})
      end
   end
end

-- dump parameters

-- This takes a single parameter, an instance id (usually the $0 of the
-- calling patch) which is used as a prefix to the receiver symbols the
-- parameters should be sent to. The id is cached so that subsequent dump
-- operations can be invoked without the id.

function raptor:in_1_dump(atoms)
   local init = self.id == nil
   local id = atoms[1] and atoms[1] or self.id
   if type(id) == "number" then
      id = string.format("%d", id)
   elseif type(id) ~= "string" then
      return
   end
   for i, param in ipairs(params) do
      local sym = string.format("%s-%s", id, param.name)
      pd.send(sym, "set", {self.param_val[i]})
   end
   -- these need special treatment
   pd.send(string.format("%s-%s", id, "tempo"), "set", {self.tempo})
   pd.send(string.format("%s-%s", id, "meter-num"), "set", {self.n})
   pd.send(string.format("%s-%s", id, "meter-denom"), "set", {self.m})
   pd.send(string.format("%s-%s", id, "division"), "set", {self.division})
   if init then
      pd.send(string.format("%s-%s", id, "preset"), "symbol", {"default"})
      if debug >= 1 then
	 print(string.format("raptor (id %d) is up and running!", id))
	 -- this prints the names of all presets in the console
	 self:in_1_preset({})
      end
      -- we keep track of the id for later preset operations
      self.id = id
   end
end

-- midi learn

function raptor:load_map()
   local fname = self._canvaspath .. "lib/midi_map.lua"
   local fp = io.open(fname, "r")
   if fp then
      local midi_map = fp:read("a")
      if midi_map then
	 local f = load("return " .. midi_map)
	 if type(f) == "function" then
	    midi_map = f()
	    -- do some quick plausability checks
	    if type(midi_map) == "table" then
	       self.midi_map = midi_map
	    end
	 end
      end
      fp:close()
   end
end

function raptor:save_map()
   -- collect garbage
   local midi_map = {}
   for cc, map in pairs(self.midi_map) do
      if map and next(map) then
	 midi_map[cc] = map
      end
   end
   self.midi_map = midi_map
   local fname = self._canvaspath .. "lib/midi_map.lua"
   local fp = io.open(fname, "w")
   if fp then
      fp:write(_inspect(self.midi_map))
      fp:close()
      -- broadcast a message to all raptor instances so that they can update
      -- themselves
      pd.send("__raptor", "midimap", {})
   end
end

function raptor:map_get(cc, ch)
   local map = self.midi_map[cc]
   if map then
      return map[ch]
   else
      return nil
   end
end

function raptor:map_set(cc, ch, var)
   local map = self.midi_map[cc]
   if not map then
      map = {}
      self.midi_map[cc] = map
   end
   map[ch] = var
end

function raptor:map_find(var)
   for cc, map in pairs(self.midi_map) do
      for ch, v in pairs(map) do
	 if v == var then
	    return cc, ch
	 end
      end
   end
   return nil, nil
end

function raptor:map_mode(status)
   self.midi_learn = status
   if self.id then
      -- report changes to the panel
      local id = self.id
      pd.send(string.format("%s-midi-learn", id), "float", {status})
   end
end

function raptor:learn()
   if self.midi_learn_cc and self.midi_learn_var then
      local var = self:map_get(self.midi_learn_cc, self.midi_learn_ch)
      self:map_set(self.midi_learn_cc, self.midi_learn_ch, self.midi_learn_var)
      self:map_mode(0)
      print(string.format("CC%d-%d %smapped to %s", self.midi_learn_cc, self.midi_learn_ch, var and "re" or "", self.midi_learn_var))
      self:save_map()
   elseif self.midi_learn_cc then
      local var = self:map_get(self.midi_learn_cc, self.midi_learn_ch)
      if var then
	 print(string.format("remapping CC%d-%d currently mapped to %s, wiggle a control", self.midi_learn_cc, self.midi_learn_ch, var))
	 print("press learn again to abort, or press unlearn to unmap")
      else
	 print(string.format("mapping CC%d-%d, wiggle a control", self.midi_learn_cc, self.midi_learn_ch))
      end
   elseif self.midi_learn_var then
      local cc, ch = self:map_find(self.midi_learn_var)
      if cc then
	 print(string.format("mapping param %s already mapped to CC%d-%d, enter CC", self.midi_learn_var, cc, ch))
	 print("press learn again to abort, or press unlearn to unmap")
      else
	 print(string.format("mapping param %s, enter CC", self.midi_learn_var))
      end
   end
end

function raptor:in_1_learn()
   if self.midi_learn == 1 then
      print("MIDI learn mode aborted")
      self:map_mode(0)
   else
      self.midi_learn_cc = nil
      self.midi_learn_ch = nil
      self.midi_learn_var = nil
      self:map_mode(1)
      print("MIDI learn mode, enter CC or wiggle a control")
      print("press learn again to abort")
   end
end

function raptor:in_1_unlearn()
   if self.midi_learn == 1 then
      local done = false
      if self.midi_learn_cc then
	 local var = self:map_get(self.midi_learn_cc, self.midi_learn_ch)
	 if var then
	    self:map_set(self.midi_learn_cc, self.midi_learn_ch, nil)
	    print(string.format("CC%d-%d unmapped", self.midi_learn_cc, self.midi_learn_ch))
	    self:save_map()
	    done = true
	 end
      elseif self.midi_learn_var then
	 local cc, ch = self:map_find(self.midi_learn_var)
	 if cc then
	    self:map_set(cc, ch, nil)
	    print(string.format("CC%d-%d unmapped", cc, ch))
	    self:save_map()
	    done = true
	 end
      end
      if not done then
	 print("MIDI learn mode aborted")
      end
      self:map_mode(0)
   else
      self.midi_learn_cc = nil
      self.midi_learn_ch = nil
      self.midi_learn_var = nil
      self:map_mode(1)
      print("MIDI learn mode, enter CC or wiggle a control")
      print("press learn again to abort")
   end
end

-- generic param setter/getter

function raptor:param(var, val)
   local i = param_i[var]
   if i then
      if val == nil then
	 -- report the current value
	 self:outlet(1, "float", {self.param_val[i]})
      elseif type(val) == "number" then
	 local v = val
	 if int_param[i] then
	    -- force integer values
	    v = math.floor(v)
	 end
	 -- clamp to the prescribed range
	 if v > params[i].max then
	    v = params[i].max
	 end
	 if v < params[i].min then
	    v = params[i].min
	 end
	 if self.param_set[i] and v ~= self.param_val[i] then
	    -- update the current value
	    self.param_val[i] = v
	    if self.param_set[i] == self.set then
	       -- these actually live in the raptor instance
	       self.param_set[i](self, var, v)
	    else
	       -- these all live in the arpeggiator
	       self.param_set[i](self.arp, v)
	    end
	    if self.id then
	       -- report changes to the panel
	       local id = self.id
	       pd.send(string.format("%s-%s", id, var), "set", {v})
	    end
	 end
      end
   end
end

function raptor:in_1(sel, atoms)
   if self.midi_learn == 1 then
      self.midi_learn_var = sel
      self:learn()
   end
   self:param(sel, atoms[1])
end
