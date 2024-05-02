
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

-- Global configuration data.

-- debug_level: This only affects the plugin code. The available levels are:
-- 1: print preset changes only, 2: also print the current beat and other
-- important state information, 3: also print note output, 4: print
-- everything, including note input. Output goes to the Pd console.
-- NOTE: To debug the internal state of the arpeggiator object, including
-- pattern changes and note generation, use the arp.debug setting below.
local debug_level = 0

-- pickup mode: When enabled (default), this makes sure that CC changes do not
-- cause large jumps if the CC value does not reflect the current value of the
-- corresponding Raptor parameter. This can happen, specifically, if the MIDI
-- mapping of a knob suddenly changes due to a new MIDI learn binding or
-- actions on the controller itself, or if the parameter was changed
-- internally or through a GUI action.
local pickup_mode = 1

-- Metronome click. This is a MIDI note number which will be sent on channel
-- 10 of the first MIDI output port. The default of 54 is the Tambourine in a
-- GM-compatible kit. You might need to adjust this for the instrument that
-- you're using.
local metro_click = 54

-- The volume of the metronome, as a fraction of the absolute velocities (1
-- means normal volume, 0 means off).
local metro_volume = 1

-- Special device support. At present, these all work together nicely, so we
-- have them all enabled by default. But you can turn them on or off
-- individually by adjusting the corresponding variables below. For further
-- details about each device, please check the documentation and the
-- corresponding MIDI map in the data subdirectory.

local launchkey = 1     -- Novation Launchkey
local launchpad = 1     -- Novation Launchpad
local launchcontrol = 1 -- Novation Launch Control XL
local midimix = 1       -- AKAI Professional MIDIMIX
local apcmini = 1       -- AKAI Professional APC mini (mk2)
local pacer = 1         -- Nektar PACER
local djcontrol = 1     -- Hercules DJ Control devices

-- APC mini control port.

-- This is detected at startup and must be 3 or 4. Only the APC mini mk2 is
-- supported at present. Also note that you can't have a Launchpad on the same
-- port, since the two are incompatible.

local apcmini_portno

-- Additional parameters for the Launchpad.

-- If you know the ids (12 = X, 13 = Mini MK3, 14 = Pro MK3) of your
-- Launchpads, you can put them below, otherwise we'll try to guess them with
-- an identity inquiry sysex at startup. NOTE: If you set this, make it a
-- table indexed by port numbers (only 3 and 4 will work at present). E.g.
-- (Pro MK3 on port 3, X on port 4):
--local launchpad_id = { [3] = 14, [4] = 12 }
local launchpad_id = nil

-- This sets the sensitivity of the pads on the launch grid. Smaller values >
-- 0 mean lighter touches will trigger; a touch below the threshold will show
-- a tooltip with the current binding in the console without triggering. A
-- zero value completely disables the launch grid, so that none of the pads
-- can be triggered or mapped using MIDI learn.
local launchpad_trigger = 30

-- Maximum number of most salient steps per bar to flash the Novation logo for
-- the rhythm display. Just set this to 0 if you hate the blinkenlights. Works
-- the same as djcontrol_n_pulses below (which see for a more elaborate
-- explanation of this parameter).
local launchpad_n_pulses = 7

-- Fader orientation: This is a table with five entries, one for each fader
-- bank (Volume, Pan, Send A, Send B, Extra; 0 means vertical, 1 horizontal).
local launchpad_fader_orientation = { 0, 1, 1, 1, 0 }

-- Display a funky startup animation (scrolling text, LP X and Mini only).
local launchpad_welcome = "Raptor 7 ready"

-- Additional parameters for the Launchkey.

-- If you know the sysex id of your Launchkey (18 = LK MK3 88, 15 = any other
-- LK MK3), you can put it below, otherwise we'll try to guess it with an
-- identity inquiry sysex at startup.
local launchkey_id = nil

-- Initial text to show on the Launchkey LCD screen (n/a on the Mini).
local launchkey_welcome = "Raptor 7 ready"

-- Additional parameters for the DJ Control.

-- This value determines how fast the playback position moves in response to
-- jog wheel movements. Larger values slow it down, smaller values speed it
-- up. The default value of 10 seems to be about right for me, but YMMV.
local djcontrol_scrub_factor = 10

-- The rhythm backlight could get rather busy with complex meters, so we only
-- trigger the n most salient pulses, as determined by the weight of the pulse
-- (using Barlow indispensabilities) and the total number of beats. The
-- default which I found to work best with most meters is 7, but you can
-- adjust that value according to your preferences below. Setting
-- djcontrol_n_pulses to a very large value like 1000 will trigger each and
-- every pulse. Decreasing the value gradually thins out the rhythm display.
-- Setting it to 0 disables the rhythm display.
local djcontrol_n_pulses = 7

-- -------------------------------------------------------------------------

-- make sure that this is set if any of the above is enabled
local have_control = launchkey ~= 0 or launchpad ~= 0 or launchcontrol ~= 0 or
   midimix ~= 0 or pacer ~= 0 or djcontrol ~= 0

-- -------------------------------------------------------------------------

-- For MIDI pass-through, we filter out MIDI data from port #2 by default, if
-- any of the control surfaces is enabled, and also from port #3 and #4, if
-- the Launchpad is enabled, since it uses these ports. This prevents control
-- surface data from "leaking" and triggering spurious notes and control
-- changes in the arpeggiator or connected synthesizers. (You know the drill
-- if you ever hooked up a DAW controller to a synthesizer.)

-- The following value is a MIDI input port number and can be changed here if
-- needed, or you can set it at runtime by sending raptor a 'thru' message.
-- Note that port #1 is never filtered out, since it's the primary input. Any
-- value n > 1 means that data coming from port 2 to n will be filtered out,
-- while data coming from ports n+1 and above will be passed through. Thus,
-- n = 0 or 1 effectively disables the filter, while n >= N (where N is the
-- total number of MIDI input ports) filters out data from all ports > 1.

-- A reasonable default is 2 if any control surface is connected, and 4, if
-- the Launchpad is, which is what we use here.
local midi_thru = not have_control and 1 or launchpad == 0 and 2 or 4

-- midimap_name: The name of the file in the data directory in which MIDI
-- bindings are stored. You can change this if you frequently switch between
-- different MIDI setups, but note that this file is modified any time you add
-- or remove a binding using raptor's MIDI learn facility.
local midimap_name = "midi.map"

-- -------------------------------------------------------------------------

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
local function inspect(x, options)
   if not options then
      options = {newline = " ", indent = ""}
   end
   return _inspect(x, options)
end

-- -------------------------------------------------------------------------

-- tie-in with data from the config patch

local first_config = {}

local function controller_setup(data)
   local id, config_launchkey, config_launchpad, config_launchcontrol, config_midimix, config_pacer, config_djcontrol, config_pickup, config_click = table.unpack(data)
   local last_state = {have_control = have_control, launchpad = launchpad}
   if not first_config[id] then
      launchkey = launchkey*config_launchkey ~= 0 and 1 or 0
      launchpad = launchpad*config_launchpad ~= 0 and 1 or 0
      launchcontrol = launchcontrol*config_launchcontrol ~= 0 and 1 or 0
      midimix = midimix*config_midimix ~= 0 and 1 or 0
      pacer = pacer*config_pacer ~= 0 and 1 or 0
      djcontrol = djcontrol*config_djcontrol ~= 0 and 1 or 0
      pickup_mode = pickup_mode*config_pickup ~= 0 and 1 or 0
      metro_click = config_click and math.floor(config_click) or metro_click
      first_config[id] = true
   else
      launchkey = config_launchkey ~= 0 and 1 or 0
      launchpad = config_launchpad ~= 0 and 1 or 0
      launchcontrol = config_launchcontrol ~= 0 and 1 or 0
      midimix = config_midimix ~= 0 and 1 or 0
      pacer = config_pacer ~= 0 and 1 or 0
      djcontrol = config_djcontrol ~= 0 and 1 or 0
      pickup_mode = config_pickup ~= 0 and 1 or 0
      metro_click = config_click and math.floor(config_click) or metro_click
   end
   have_control = launchkey ~= 0 or launchpad ~= 0 or launchcontrol ~= 0 or
      midimix ~= 0 or pacer ~= 0 or djcontrol ~= 0
   if last_state.have_control ~= have_control or
      last_state.launchpad ~= launchpad then
      -- reset the MIDI thru settings
      midi_thru = not have_control and 1 or launchpad == 0 and 2 or 4
   end
end

local function launchpad_setup(data)
   local id, config_launchpad_trigger, config_launchpad_n_pulses = table.unpack(data)
   launchpad_trigger = math.floor(config_launchpad_trigger)
   launchpad_n_pulses = math.floor(config_launchpad_n_pulses)
end

local function djcontrol_setup(data)
   local id, config_djcontrol_scrub_factor, config_djcontrol_n_pulses = table.unpack(data)
   djcontrol_scrub_factor = math.floor(config_djcontrol_scrub_factor)
   djcontrol_n_pulses = math.floor(config_djcontrol_n_pulses)
end

-- -------------------------------------------------------------------------

-- Helper functions: ASA note names

-- We use the most likely spellings here, but of course this will depend
-- on the key you're in, so feel free to change this as wanted.

local notename = {"C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "C"}

local function asa_pitch(n, ch)
   n = math.floor(n)
   local pc, oct = n % 12, n // 12
   -- using ASA standard octave numbering
   if ch then
      return string.format("%s%d-%d", notename[pc+1], oct-1, ch)
   else
      return string.format("%s%d", notename[pc+1], oct-1)
   end
end

-- CC descriptions

local function cc_name(cc, ch)
   if ch then
      return string.format("CC%d-%d", cc, ch)
   else
      return string.format("CC%d", cc)
   end
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
	 latch = nil, down = -1, up = 1, mode = 0, raptor = 0,
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
   self.down, self.up, self.mode, self.raptor = -1, 1, 0, 0
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
      local a, b = p // self.beats + 1, n // self.beats
      self.loop_counter = {p, a, b}
      if p % self.beats == 0 then
	 --print(string.format("loop: playing bar %d/%d", a, b))
      end
   end
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
      local loop, n = {}, math.min(#self.loop, self.loopsize)
      -- make sure to keep meter and tempo information if we have it
      loop.meter = self.loop.meter
      loop.tempo = self.loop.tempo
      -- shorten the table to the current loop size if needed
      table.move(self.loop, 1, n, 1, loop)
      -- add some pretty-printing
      local function bars(level, count)
	 if level == 1 and count%self.beats == 0 then
	    return string.format("-- bar %d", count//self.beats+1)
	 end
      end
      local function notes(level, count)
	 if level == 1 then
	    local ns = loop[count][1]
	    if type(ns) == "number" then
	       ns = {ns}
	    elseif type(ns) == "table" and next(ns) then
	       -- make sure that we take a copy here
	       ns = {table.unpack(ns)}
	    else
	       return
	    end
	    for i = 1, #ns do
	       ns[i] = asa_pitch(ns[i])
	    end
	    return string.format("-- %s", table.concat(ns, ", "))
	 end
      end
      f:write(string.format("-- saved by Raptor %s\n", os.date()))
      f:write(inspect(loop, {extra = 1, addin = bars, addout = notes}))
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
	    return "loopsize", self.loopsize
	 end
      end
   elseif cmd == 2 then
      -- check that file exists, report result
      return "loopcheck", fexists(path) and 1 or 0
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
      self.loopdir = x[1]
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
   self.loop_counter = nil
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

-- change the current loop index
function arpeggio:set_loopidx(x)
   x = self:intarg(x)
   if type(x) == "number" and self.loopstate == 1 and self.loopidx ~= x then
      self.loopidx = math.max(0, x) % math.max(1, math.min(#self.loop, self.loopsize))
      self.idx = self.loopidx % self.beats
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

-- The id of the global time master. This is shared between all instances.
local time_master

-- Parameter and preset tables. These are mostly the same as in the Ardour
-- plugin. Note that some of the fields aren't used in the Pd implementation,
-- which adds a few special flags of its own, see below.

local hrm_scalepoints = { ["0.09 (minor 7th and 3rd)"] = 0.09, ["0.1 (major 2nd and 3rd)"] = 0.1, ["0.17 (4th)"] = 0.17, ["0.21 (5th)"] = 0.21, ["1 (unison, octave)"] = 1 }

-- Special flags not in the Ardour plugin:

-- noload: performance controls to be skipped when loading and saving presets

-- time, transport: time- and transport-related controls (the former live in
-- the panel, the latter in the time subpatch; also note that with the
-- exception of division, these controls aren't in the Ardour version, because
-- they are maintained in the DAW)

-- looper: looper controls

local params = {
   { type = "input", name = "bypass", min = 0, max = 1, default = 0, toggled = true, noload = true, doc = "bypass the arpeggiator, pass through input notes" },
   { type = "input", name = "division", min = 1, max = 7, default = 1, integer = true, noload = true, time = true, doc = "number of subdivisions of the beat" },
   -- These aren't in the Ardour plugin, as meter and tempo get set through
   -- the DAW's timeline, but it's useful to have these values as parameters
   -- in the stand-alone version, so that they can be mapped via MIDI learn.
   { type = "input", name = "meter-num", min = 1, max = 16, default = 4, integer = true, noload = true, time = true, doc = "number of beats per bar" },
   { type = "input", name = "meter-denom", min = 1, max = 16, default = 4, integer = true, noload = true, time = true, doc = "note value of the beat" },
   { type = "input", name = "tempo", min = 0, max = 240, default = 120, integer = true, noload = true, time = true, doc = "tempo (bpm)" },
   { type = "input", name = "pgm", min = 0, max = 128, default = 0, integer = true, doc = "program change", scalepoints = { default = 0 } },
   { type = "input", name = "latch", min = 0, max = 1, default = 0, toggled = true, noload = true, doc = "toggle latch mode" },
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
   { type = "input", name = "loop", min = 0, max = 1, default = 0, toggled = true, noload = true, doc = "toggle loop mode" },
   { type = "input", name = "mute", min = 0, max = 1, default = 0, toggled = true, noload = true, doc = "turn the arpeggiator off, suppress all note output" },
   { type = "input", name = "play", min = 0, max = 1, default = 0, toggled = true, noload = true, transport = true, doc = "start or stop playback" },
   { type = "input", name = "pulse", min = 0, max = 1, default = 0, toggled = true, noload = true, transport = true, doc = "trigger pulses manually" },
   { type = "input", name = "pos", min = -24, max = 24, default = 0, integer = true, noload = true, transport = true, doc = "anacrusis control" },
   { type = "input", name = "rewind", min = 0, max = 1, default = 0, toggled = true, noload = true, transport = true, doc = "rewind (relocate the playhead to the anacrusis)" },
   -- synthetic looper commands
   { type = "input", name = "loop-load", min = 0, max = 1, default = 0, toggled = true, noload = true, looper = true, doc = "load loop file" },
   { type = "input", name = "loop-save", min = 0, max = 1, default = 0, toggled = true, noload = true, looper = true, doc = "save loop file" },
   { type = "input", name = "loop-prev", min = 0, max = 1, default = 0, toggled = true, noload = true, looper = true, doc = "previous loop" },
   { type = "input", name = "loop-next", min = 0, max = 1, default = 0, toggled = true, noload = true, looper = true, doc = "next loop" },
   -- metronome click
   { type = "input", name = "click", min = 0, max = 1, default = 0, toggled = true, noload = true, transport = true, doc = "toggle the metronome click" },
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
   local last_click = self.click
   local last_play = self.play
   local last_pulse = self.pulse
   local last_pos = self.pos
   local last_rewind = self.rewind
   local last_n = self.n
   local last_division = self.division
   local last_inchan = self.inchan
   local last_chan = self.chan
   local last_pgm = self.pgm
   if param == "meter-num" then
      param = "n"
   elseif param == "meter-denom" then
      param = "m"
   end
   self[param] = x
   -- various state changes that need special treatment
   if (self.bypass ~= last_bypass and self.bypass ~= 0) or
      (self.mute ~= last_mute and self.mute ~= 0) then
      -- turn off any sounding notes from the arpeggiator
      self:notes_off()
   end
   if self.mute ~= last_mute then
      -- djcontrol tie-in, updates the MUTE (PFL) buttons
      self:djcontrol_mute(self.mute)
   end
   if self.n*self.division ~= last_n*last_division then
      -- update the meter
      self:update_meter()
   end
   -- calculate the new delay (note-off time) in ms
   local delay = 60000/self.tempo * 4/self.m/self.division
   if self.last_delay and delay ~= self.last_delay then
      -- we want to update the delay time immediately if it has changed, so
      -- that we don't get stuck waiting for the next pulse if the previous
      -- delay time was very large or even infinite (tempo = 0)
      self:outlet(4, "float", { delay })
   end
   if self.inchan ~= last_inchan and self.inchan > 0 then
      -- change of input channel, kill off chord memory and stop notes
      self.arp:panic()
      self:notes_off()
   end
   if self.pgm ~= last_pgm or self.chan ~= last_chan or
      self:get_chan(self.chan) ~= self.chan then
      -- program or output channel has changed, send the program change
      self.chan = self:get_chan(self.chan)
      if self.pgm > 0 then
	 self:outlet(1, "pgm", { self.pgm, self.chan })
      end
   end
   -- transport
   local tmaster = self:check_master()
   local master_check = self.assert_master or tmaster
   -- we have to go to some lengths here to deal with the djcontrol which
   -- may set transport parameters independently for each deck
   if self.play ~= last_play then
      if master_check then
	 if not tmaster and self.transport ~= 0 then
	    -- if we're not the real time master and transport is rolling,
	    -- tell the old master to hand over at the next pulse in order to
	    -- not disrupt playback (djcontrol; this can only happen if the
	    -- PLAY button was clicked on the other deck)
	    pd.send(string.format("%s-%s", time_master, "new-master"), "float", {tonumber(self.id)})
	 else
	    -- we're the real time master, or transport is stopped; just
	    -- start/stop the playback
	    pd.send(string.format("%s-%s", self.id, "play"), "float", {self.play})
	 end
      end
   end
   if self.pos ~= last_pos and master_check then
      pd.send(string.format("%s-%s", self.id, "pos"), "float", {self.pos})
   elseif self.pos ~= last_pos and not master_check then
      -- if we're not the time master, we still need to update the anacrusis
      -- the difference is that this change isn't announced to all raptors
      pd.send(string.format("%s-%s", self.id, "pos"), "set", {self.pos})
      pd.send(string.format("%s-%s", self.id, "posvar"), "float", {self.pos})
   end
   if self.rewind ~= last_rewind and self.rewind >= 0 and master_check then
      pd.send(string.format("%s-%s", self.id, "rewind"), "bang", {})
   end
   if self.pulse ~= last_pulse and self.pulse >= 0 and self.id then
      pd.send(string.format("%s-%s", self.id, "pulse"), "bang", {})
   end
   if self.click ~= last_click and self.id then
      pd.send(string.format("%s-%s", self.id, "click"), "set", {self.click})
   end
end

function raptor:set_param_tables()
   -- this initializes the parameter setter callbacks; this needs to be redone
   -- after reloading the object (pdx.reload)
   self.param_set = { self.set, self.set, self.set, self.set, self.set, self.set, self.arp.set_latch, self.arp.set_up, self.arp.set_down, self.set, self.arp.set_mode, self.arp.set_raptor, self.arp.set_minvel, self.arp.set_maxvel, self.arp.set_velmod, self.arp.set_gain, self.arp.set_gate, self.arp.set_gatemod, self.arp.set_wmin, self.arp.set_wmax, self.arp.set_pmin, self.arp.set_pmax, self.arp.set_pmod, self.arp.set_hmin, self.arp.set_hmax, self.arp.set_hmod, self.arp.set_pref, self.arp.set_prefmod, self.arp.set_smin, self.arp.set_smax, self.arp.set_smod, self.arp.set_nmax, self.arp.set_nmod, self.arp.set_uniq, self.arp.set_pitchhi, self.arp.set_pitchlo, self.arp.set_pitchtracker, self.set, self.set, arp_set_loopsize, self.arp.set_loop, self.set, self.set, self.set, self.set, self.set, self.set, self.set, self.set, self.set, self.set }
end

-- table of the ids of all running raptor instances
raptor.instances = {}
-- assigned deck information
raptor.decks = {}
-- current presets
raptor.presets = {}

function raptor:get_instance(id1)
   if not id1 then
      id1 = self.id
   end
   if id1 then
      for i, id2 in ipairs(raptor.instances) do
	 if id2 == id1 then
	    return i
	 end
      end
   end
   return 0 -- indicates not found or id not set
end

-- out-of-band event signaling, to circumvent finalization issues where our
-- outlets might be gone already
function raptor:out(i, sel, atoms)
   local id = self.id
   if id then
      pd.send(string.format("%s-out%d", id, i), sel, atoms)
   else
      -- if the id hasn't been set yet, early on during initialization, it
      -- should still be safe to output directly through the outlets
      self:outlet(i, sel, atoms)
   end
end

function raptor:initialize(sel, atoms)
   pdx.reload(self)

   self.inlets = 2
   self.outlets = 4

   -- initialize param values
   self.param_val = {}
   for i = 1, n_params do
      self.param_val[i] = params[i].default
   end

   -- these are maintained in the Pd object
   self.bypass = 0
   self.mute = 0
   self.click = 0

   -- default meter (numerator, denominator, and subdivision) and tempo
   self.n = 4
   self.m = 4
   self.division = 1
   self.tempo = 120

   -- transport
   self.transport = 0
   self.play = 0
   self.pulse = 0
   self.pos = 0
   self.rewind = 0

   -- create the arpeggiator (default meter)
   self.arp = arpeggio:new(self.n * self.division)

   -- set the base directory for the looper
   self.arp:set_loopdir(self._canvaspath)

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
   self.pgmset = false
   self.pgm = 0
   self.inchan = 0
   self.outchan = 0
   self.chan = 1
   self.backup_chan = 1
   self.transp = 0

   -- state of auxiliary control surfaces
   self.shift = false
   self.thru = midi_thru
   -- djcontrol state
   self.stopped = false
   self:djcontrol_init()
   -- launchpad
   self.lp_fader_map = {}
   self.lp_fader_val = {}
   self.lp_alt_cc = {}
   self.launchpad_faders = {}
   self.launchpad_page = {}
   self.launchpad_drums = {}

   -- midi learn
   self.midi_map = {}
   self.midi_learn = 0
   self.midi_learn_cc = nil
   self.midi_learn_ch = nil
   self.midi_learn_var = nil
   self.midi_learn_val = nil
   self.midi_learn_pol = nil
   self.midi_learn_tgl = nil
   self:load_map()

   -- ccmaster is a flag which indicates whether we're responding to mapped
   -- MIDI CC. This is nil (indicating omni mode) by default, but can be
   -- changed to a single raptor instance with the mastercc message.
   self.ccmaster = nil

   -- deck is the assigned deck number. This is meant for DJ controllers,
   -- which usually have two or more identical sets of controls. It is 0
   -- ("omni") by default, but can be changed with the 'deck' message, or the
   -- corresponding control in the init subpatch of the main patch.

   -- NOTE: Normally, this doesn't have any effect, unless a special
   -- controller tie-in uses this number to filter out messages based on the
   -- deck number (see djcontrol for an example).
   self.deck = 0

   -- instance id; this gets initialized later by the dump method, see below
   self.id = nil

   -- initialize the user presets
   self:load_presets()
   self.presetno = 1

   -- create a global receiver, so that we can tell all instances about global
   -- state changes
   self.recv = pd.Receive:new():register(self, "__raptor", "receive")

   -- initialize the note-off timer
   self.clock = pd.Clock:new():register(self, "notes_off")

   -- initialize the launchpad fader timers
   self.launchpad_momentary = {}
   self.launchpad_clock = {}
   self.launchpad_clock[3] = pd.Clock:new():register(self, "launchpad_fader_timer_cb3")
   self.launchpad_clock[4] = pd.Clock:new():register(self, "launchpad_fader_timer_cb4")

   -- this fires once, some time *after* the driver initialization timer, to
   -- complete the Launchpad and APCmini initializations
   self.idreq_clock = pd.Clock:new():register(self, "idreq_timer_cb")

   -- this also fires once, to complete the Launchkey initializations
   self.init2_clock = pd.Clock:new():register(self, "late_init2")

   -- initialize and kick off the driver initialization timer
   self.init_clock = pd.Clock:new():register(self, "late_init")
   self.init_clock:delay(500)

   return true
end

local idreq_check

function raptor:idreq_timer_cb()
   -- this will be checked by whatever instance gets here first
   if not idreq_check then
      -- disable the Launchpad driver if we didn't get the expected reply
      if launchpad ~= 0 then
	 if launchpad_id and next(launchpad_id) then
	    for portno, id in pairs(launchpad_id) do
	       print(string.format("Launchpad %s connected on port #%d", self:launchpad_model_name(id), portno))
	    end
	    self:launchpad_init()
	 else
	    print("No known Launchpad device detected, driver disabled")
	    launchpad = 0
	 end
      end
      -- check for the APCmini
      if apcmini ~= 0 then
	 if apcmini_portno then
	    if apcmini_portno ~= 3 and apcmini_portno ~= 4 then
	       -- invalid port
	       print(string.format("APC mini: wrong port #%d, driver disabled", apcmini_portno))
	       apcmini = 0
	    elseif launchpad_id and launchpad_id[apcmini_portno] then
	       -- Launchpad connected port
	       print(string.format("APC mini: Launchpad on same port #%d, driver disabled", apcmini_portno))
	       apcmini = 0
	    else
	       print(string.format("APC mini connected on port #%d", apcmini_portno))
	       self:apcmini_init()
	    end
	 else
	    print("No APC mini device detected, driver disabled")
	    apcmini = 0
	 end
      end
      idreq_check = true
   end
end

function raptor:late_init()
   if not launchpad_id or not apcmini_portno then
      -- device inquiry message (we'll pick up the result later)
      for portno = 3, 4 do
	 self:out(2, "float", {portno})
	 self:out(1, "sysex", {126, 127, 6, 1})
      end
      self.idreq_clock:delay(500)
   end
   -- APC mini initialization
   if apcmini_portno then
      self:apcmini_init()
   end
   -- launchpad initialization
   if launchpad_id then
      self:launchpad_init()
   else
      -- initialize launchpad_id
      launchpad_id = {}
   end
   -- launchkey initialization
   if not launchkey_id then
      -- device inquiry message (we'll pick up the result later)
      self:out(2, "float", {2})
      self:out(1, "sysex", {126, 127, 6, 1})
   end
   self:launchkey_init()
   -- djcontrol initialization
   self:djcontrol_state_init()
   -- kick off the timer for even later initializations
   self.init2_clock:delay(600)
end

function raptor:late_init2()
   -- launchkey initialization, part 2
   self:launchkey_init2()
   -- launchpad initialization, part 2 (this needs to run after the idreq timer)
   if launchpad ~= 0 then
      -- still need to initialize the device select buttons
      self:launchpad_ccmaster(0)
   end
end

function raptor:check_master()
   return time_master and self.id == time_master
end

function raptor:check_ccmaster(var)
   if not self.ccmaster or self.ccmaster == self.id then
      -- omni mode or we're the ccmaster
      return true
   elseif var then
      local i = param_i[var]
      -- also check for time and transport parameters, we need to make sure
      -- that these reach the time master
      return i and (params[i].time or params[i].transport)
   else
      return false
   end
end

function raptor:finalize()
   for i = 3, 4 do
      self.launchpad_clock[i]:destruct()
   end
   self.idreq_clock:destruct()
   self.init2_clock:destruct()
   self.init_clock:destruct()
   self.clock:destruct()
   self.recv:destruct()
   self:launchpad_fini()
   self:launchkey_fini()
   self:apcmini_fini()
   self:djcontrol_state_fini()
   if self.ccmaster and self:check_ccmaster() then
      -- tell all running raptors that we're back to omni
      pd.send("all-arp", "ccmaster", {0, self.id})
   end
   local i = self:get_instance()
   if i > 0 then
      -- update the instance selection feedback on devices which need it
      local k = #raptor.instances
      for j = i+1, k do
	 local state = raptor.instances[j] == self.ccmaster and 1 or 0
	 local deck = raptor.decks[raptor.instances[j]]
	 self:launchpad_ccmaster(state, j-1)
	 self:launchkey_ccmaster_state(state, j-1)
	 self:djcontrol_ccmaster(state, j-1, deck)
      end
      self:launchpad_ccmaster(0, k, 0)
      self:launchkey_ccmaster_state(0, k, 0)
      local deck = raptor.decks[raptor.instances[k]]
      self:djcontrol_ccmaster(0, k, deck)
      self:launchcontrol_ccmaster(0)
      self:midimix_ccmaster(0)
      self:apcmini_ccmaster(0)
      -- remove ourself from the instances table
      table.remove(raptor.instances, i)
      -- also remove the assigned deck and preset information
      raptor.decks[self.id] = nil
      raptor.presets[self.id] = nil
   end
end

-- controller setup

function raptor:in_1_config(data)
   local last_apcmini = apcmini
   local last_launchkey = launchkey
   local last_launchpad = launchpad
   controller_setup(data)
   if last_launchpad ~= launchpad then
      -- process launchpad status change
      if launchpad == 0 then
	 self:launchpad_fini(true)
      else
	 self:launchpad_init()
      end
   end
   if last_launchkey ~= launchkey then
      -- process launchkey status change
      if launchkey == 0 then
	 self:launchkey_fini(true)
      else
	 self:launchkey_init()
      end
   end
   if last_apcmini ~= apcmini then
      -- process APC mini status change
      if apcmini == 0 then
	 self:apcmini_fini(true)
      else
	 self:apcmini_init()
      end
   end
end

function raptor:in_1_lpconfig(data)
   launchpad_setup(data)
end

function raptor:in_1_djconfig(data)
   djcontrol_setup(data)
end

-- pulses

function raptor:notes_off()
   if self.last_notes then
      -- kill the old notes
      for _, num in ipairs(self.last_notes) do
	 if debug_level >= 3 then
	    print(string.format("[out] note off %d", num))
	 end
	 self:outlet(1, "note", { num, 0, self.last_chan })
      end
      self.last_notes = nil
      -- stop the note-off timer in case it's still pending
      self.clock:unset()
   end
end

function raptor:metro_click(w, val)
   -- metronome click
   if self:check_master() then
      local num = metro_click
      if w == 0 and val == 0 then
	 -- turn the metronome click off
	 self:out(1, "note", {num, 0, 10})
      elseif self.click ~= 0 then
	 -- turn off the previous click
	 self:out(1, "note", {num, 0, 10})
	 -- w is the weight, val the velocity, n the number of beats per bar
	 -- to trigger, b the total number of beats. NOTE: We borrow the
	 -- launchpad_n_pulses variable from the Launchpad config here, so
	 -- that the metronome click is always in sync with the Launchpad's
	 -- pulse display.
	 local n, b = launchpad_n_pulses, self.arp.beats
	 local state = w >= b-n and 1 or 0
	 local vel = math.floor(val*state*metro_volume)
	 if vel > 0 then
	    -- the next click is due
	    self:out(1, "note", {num, vel, 10})
	 end
      end
   end
end

function raptor:in_1_bang()
   -- check if we're stopped then we bail out immediately (djcontrol)
   if self.stopped then
      self:notes_off()
      return
   end
   -- grab some notes from the arpeggiator
   local p = self.arp.idx
   local notes, vel, gate, w, n = self.arp:pulse()
   -- calculate the current delay (note-off time) in ms
   local delay = 60000/self.tempo * 4/self.m/self.division
   self.last_delay = delay
   -- output the delay time until the next pulse is due on outlet #3
   self:outlet(4, "float", { delay })
   -- output the current pulse number and number of beats on outlet #2
   self:outlet(3, "list", { p, n })
   -- djcontrol tie-in, flashes the "energy" led on the encoder
   self:djcontrol_pulse(w, vel)
   -- launchpad tie-in, flashes the Novation logo
   self:launchpad_pulse(w, vel)
   -- metronome click
   self:metro_click(w, vel)
   -- check if we're bypassed or muted
   if self.bypass ~= 0 or self.mute ~= 0 then
      return
   end
   if debug_level >= 2 then
      -- print some debugging information: fractional beat number, current
      -- meter, current tempo
      print (string.format("%g - %d/%d - %g bpm",
			   math.floor(p/self.division*1000)/1000,
			   self.n, self.m, self.tempo))
   end
   -- Make sure that the gate is clamped to the 0-1 range, since we don't
   -- support overlapping notes in the current implementation.
   gate = math.max(0, math.min(1, gate))
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
      -- output the notes on outlet #1
      for i = 1, #notes do
	 local num = notes[i]+self.transp -- apply transposition
	 notes[i] = num
	 if debug_level >= 3 then
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
      if debug_level >= 2 then
	 -- monitor memory usage of the Lua interpreter
	 print(string.format("mem: %0.2f KB", collectgarbage("count")))
      end
   end
   -- provide feedback to the looper, if any
   if self.arp.loop_counter then
      self:outlet(1, "loopcounter", self.arp.loop_counter)
   end
end

-- (re)set the pulse index

function raptor:in_1_float(p)
   if type(p) == "number" then
      p = math.floor(p)
      self.arp:set_idx(p % self.arp.beats)
      if p == self.pos then
	 -- kludge: transport may trigger a "pos" (SPP) update even before the
	 -- "play" event arrives, and "play" or "rewind" may also trigger a
	 -- "pos" event afterwards; we don't want that event to be recorded if
	 -- the value hasn't changed at all, in order to not confuse MIDI
	 -- learn about which event is to be mapped
	 return
      end
      -- synthetic pos param, this can be MIDI-mapped
      self:in_1("pos", {p})
   end
end

function raptor:in_1_reset()
   self.arp:set_idx(0)
end

-- Set the pulse index directly, without synchronizing different instances.
-- This is most useful when performing with a DJ controller (see djcontrol
-- below for an example).

function raptor:set_pos(p)
   p = math.floor(p)
   self.arp:set_idx(p % self.arp.beats)
   if p ~= self.pos then
      self.pos = p
      -- We need to update the internal state manually here, without going
      -- through raptor:param() and raptor:set() which would announce the
      -- change to all Raptor instances.
      -- Update the internal param storage...
      local i = param_i["pos"]
      self.param_val[i] = p
      -- ... and the Launchpad fader bank ...
      self:launchpad_fader_val("pos", p)
      -- ... and the GUI
      if self.id then
	 pd.send(string.format("%s-%s", self.id, "pos"), "set", {p})
	 pd.send(string.format("%s-%s", self.id, "posvar"), "float", {p})
      end
   end
end

function raptor:do_rewind(p)
   p = math.floor(p)
   self.arp:set_idx(p % self.arp.beats)
   if self.id then
      pd.send(string.format("%s-%s", self.id, "do-rewind"), "bang", {})
   end
end

-- panic -- this resets the arpeggiator and stops all sounding notes

function raptor:in_1_panic()
   self.arp:panic()
   self:in_1_stop()
end

-- stop -- this just stops all sounding notes, but keeps the arpeggiator state

function raptor:in_1_stop()
   self:notes_off()
   -- turn off the pulse displays
   self:djcontrol_pulse(0, 0)
   self:launchpad_pulse(0, 0)
   -- also turn off the metronome click
   self:metro_click(0, 0)
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

function raptor:get_preset(preset)
   if type(preset) == "number" and math.floor(preset) == preset then
      local i = math.floor(preset)
      if raptor_presets[i] or self.user_presets[i-n_presets] then
	 return i
      end
   elseif type(preset) == "string" then
      -- first scan the user presets so that these can overide factory presets
      -- with the same name
      local i = self.user_preset_i[preset]
      if i then
	 return i+n_presets
      end
      i = preset_i[preset]
      if i then
	 return i
      end
   end
   return nil
end

-- Update all relevant feedback state in the (Launchpad, Launchkey, APC mini)
-- drivers after changes that might affect values and/or setup of knobs/faders
-- and the launch grid. (Assume setup changes if remap == true.)
function raptor:update_state(remap)
   -- Launchpad fader pages (this doesn't actually generate any feedback on
   -- the spot, this is deferred until the pages are shown).
   self:launchpad_update_pages()
   if remap then
      -- These all generate actual feedback data, so we only want to do this
      -- if necessary, i.e., if there might be any setup changes affecting the
      -- parameters assigned to pads and knobs.
      self:launchpad_iter(function(ch, portno) self:launchpad_pads(portno) end)
      self:launchkey_pads()
      self:launchkey_knobs()
      self:launchkey_faders()
      self:apcmini_pads()
   end
end

function raptor:recall_preset(i)
   i = self:get_preset(i)
   if not i then return end
   local preset = i > n_presets and self.user_presets[i-n_presets] or raptor_presets[i]
   if not preset then return end
   if debug_level >= 1 then
      print(string.format("preset #%d: %s", i, preset.name))
   end
   local function check(var, val)
      local i = param_i[var]
      if not i or params[i].noload then
	 return false
      elseif var == "loopsize" and self.arp.loopstate == 1 then
	 -- avoid thrashing the loop size if we're currently playing a loop
	 return false
      elseif var == "outchan" and val == 0 and self.outchan == 10 then
	 -- as an exception to the following rule, avoid being stuck on the GM
	 -- drum channel
	 return true
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
   local function set(var, val)
      if check(var, val) then
	 --print(string.format("%s = %s", var, tostring(val)))
	 self:param(var, val)
	 if self.id then
	    -- grab the value again, as raptor:param() might have updated it
	    local i = param_i[var]
	    if i then
	       val = self.param_val[i]
	       -- send the parameter so that it can be picked up by the panel
	       pd.send(string.format("%s-%s", self.id, var), "set", {val})
	    end
	 end
      end
   end
   -- LP/LK feedback state
   self:update_state()
   if not preset.params["outchan"] then
      -- force to 0
      set("outchan", 0)
   else
      -- make sure that the output channel gets set before any program change
      set("outchan", preset.params.outchan)
   end
   for var, val in pairs(preset.params) do
      if var ~= "outchan" then -- already set above
	 set(var, val)
      end
   end
   self.presetno = i
   if self.id then
      raptor.presets[self.id] = preset.name
      pd.send(string.format("%s-%s", self.id, "preset"), "symbol", {preset.name})
      pd.send(string.format("%s-%s", self.id, "presetno"), "set", {i-1})
   end
   self:launchkey_preset(preset.name)
end

function raptor:in_1_preset(atoms)
   if type(atoms[1]) == "number" or type(atoms[1]) == "string" then
      self:recall_preset(atoms[1])
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
	 if not param.noload then
	    preset.params[param.name] = self.param_val[i]
	 end
      end
      table.insert(self.user_presets, preset)
      local i = #self.user_presets
      self.user_preset_i[name] = i
      i = i+n_presets
      print(string.format("saved preset #%d: %s", i, name))
      -- write the new preset to the preset file
      local fname = self._canvaspath .. "data/presets"
      local fp = io.open(fname, "a")
      fp:write(inspect(preset), "\n")
      fp:close()
      self.presetno = i
      if self.id then
	 pd.send(string.format("%s-%s", self.id, "presetno"), "set", {i-1})
      end
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
   local fname = self._canvaspath .. "data/presets"
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

-- Special controller support. We only bind the shift button (depending on the
-- device) and the ccmaster_next, ccmaster_prev, and ccmaster_set functions
-- here. Everything else should go into the corresponding MIDI map.

-- NOTE: We always assume controllers to be connected to Pd's MIDI input port
-- #2 (or port #3, in the case of the Launchpad), so that the note and CC
-- messages from the device don't interfere with messages from the primary
-- MIDI input devices on port #1 (typically a MIDI keyboard, pad controller,
-- or other note input device). Therefore the Pd MIDI channel that we're
-- listening to is something like 16+n (32+n for the Launchpad) where n is the
-- actual MIDI channel number(s) of the device.

-- Launchpad (utilizes the Launchpad's session mode)

-- colors (see the color palette in the prog manual)
local blank, assigned = 0, 36 -- unassigned/assigned pads on the launch grid
local accent_arrows = 45 -- arrow buttons
local accent_loop = 57 -- loop buttons
-- mixer buttons
-- LPPro: Record Arm, Mute, Solo, Volume, Pan, Sends, Device, Stop Clip
local lppro_colors = {5, 61, 21, 60, 62, 64, 66, 33}
-- X/Mini: Volume, Pan, Send A, Send B, Stop Clip, Mute, Solo, Record Arm
local lpmini_colors = {60, 62, 64, 66, 33, 61, 21, 5}

-- arrow buttons: up, down, left, right
local lppro_arrow_buttons = { 80, 70, 91, 92 } or { 91, 92, 93, 94 }
local lpmini_arrow_buttons = { 91, 92, 93, 94 }

-- buttons for the fader pages (we support both the LP Pro and the Mini/X
-- buttons in parallel, they will *both* work on the LP Pro)
local lppro_fader_buttons = { 4, 5, 6, 7, 8 }
local lpmini_fader_buttons = { 89, 79, 69, 59, 49 }

local lppro_fader_pages = {}
local lpmini_fader_pages = {}
for k, i in ipairs(lppro_fader_buttons) do lppro_fader_pages[i] = k-1 end
for k, i in ipairs(lpmini_fader_buttons) do lpmini_fader_pages[i] = k-1 end

-- CC base numbers for the five fader banks 0-4 that we maintain.

-- The first four banks: Volume, Pan, Send A, Send B (Device on the Pro). On
-- the Pro, we actually have all these as preconfigured fader banks in memory,
-- on the other Launchpads they have to be set up on the fly. The controls of
-- these are mapped out in a way that is compatible with the Launch Control XL
-- factory preset 1: b = 0 = Volume, 1 = Pan (bipolar), 2 = Send A (Send on
-- the Pro), 3 = Send B (Device on the Pro).

-- The fifth "extra" fader bank (b = 4) is available on the fifth scene launch
-- button in Session mode (labeled "Probability" on the Pro and "Stop Clip" on
-- the X). This emulates the "Device" fader bank of the Launchkey, which
-- provides quick access to some frequently used controls. On the Pro, since
-- only four banks can be kept in memory, we swap out the first fader bank to
-- generate the extra one on the fly.

local lp_fader_banks = { 76, 48, 12, 28, 20 }

local function lp_fader_bank(b)
   local num = lp_fader_banks[b+1]
   if not num then
      num = lp_fader_banks[5] -- default, to be on the safe side
   end
   return num+1
end

function raptor:launchpad_fader_bank_setup(portno, b, color)
   local id = launchpad_id[portno]
   -- sets up a single fader bank on the given (launchpad) port
   local j0 = lp_fader_bank(b)
   local v = color[b+1]
   -- sysex header: identification, command 1 (fader bank setup), bank index
   -- (always zero on Mini/X), orientation (0 means vertical, 1 horizontal)
   local b0 = id==14 and b<4 and b or 0
   local orientation = launchpad_fader_orientation[b+1]
   local syx = { 0, 32, 41, 2, id, 1, b0, orientation }
   -- It seems tidier (and is likely faster) if we assemble a sysex with all
   -- faders in memory, rather than sending a sysex for each individual fader.
   for i = 0, 7 do
      local j = j0 + i
      -- There are some bipolar controls on these pages, we tie in with MIDI
      -- map and params table to check this and deal with them on the spot.
      local function bipolar_check(cc)
	 local var = self:map_get(cc, 37)
	 local i = var and param_i[var] or nil
	 local p = i and params[i] or nil
	 return p and not p.toggled and p.min == -p.max
      end
      local p = bipolar_check(j) and 1 or 0 -- 0 = unipolar, 1 = bipolar
      -- i is the fader index, p the polarity, j the CC number, v the color
      local fader = { i, p, j, v }
      -- concatenate the fader to the sysex
      table.move(fader, 1, 4, #syx+1, syx)
   end
   self:outlet(1, "sysex", syx)
end

-- shared transport state, needed for transport feedback (play button)
local rolling = 0

-- status of the welcome message, per port
local lp_welcome = {}

-- forward declaration for the feedback, since we already need this variable
-- in launchpad_fini and launchpad_master_change below
local launchpad_master = nil

function raptor:launchpad_init()
   if launchpad ~= 0 and self:check_master() then
      -- iterate over all connected launchpads
      for portno, id in pairs(launchpad_id) do
	 local ch = portno==3 and 33 or portno==4 and 49
	 -- assert ch, but to be on the safe side...
	 if not ch then goto skip end
	 local drumch = ch+8
	 -- switch the Launchpad into DAW/session mode
	 self:out(2, "float", {portno})
	 self:out(1, "sysex", {0, 32, 41, 2, id, 16, 1})
	 -- request the current layout, so that we get a sane default for
	 -- self.launchpad_page
	 self:out(1, "sysex", {0, 32, 41, 2, id, 0})
	 -- light up all buttons
	 local color = {accent_loop, 3, 1, 1, 1, 32, accent_arrows, accent_arrows}
	 for i = 1, 8 do
	    -- left- and rightmost columns (the former is only on the LPPro)
	    if id == 14 then
	       self:out(1, "ctl", {color[i], 10*i, ch})
	       self:out(1, "ctl", {lpmini_colors[9-i], 10*i+9, ch})
	    else
	       -- LP X/Mini (these run from top to bottom, so we need to
	       -- reverse the table on the fly)
	       self:out(1, "ctl", {lpmini_colors[9-i], 10*i+9, ch})
	    end
	 end
	 if id == 12 then
	    -- Capture MIDI button on the Launchpad X
	    self:out(1, "ctl", {accent_loop, 98, ch})
	 end
	 if id == 14 then -- LPPro only
	    for i = 1, 8 do
	       -- lower bottom row
	       self:out(1, "ctl", {lppro_colors[i], i, ch})
	       -- upper bottom row
	       self:launchpad_ccmaster(0, i, 0)
	    end
	 end
	 -- arrow buttons
	 local lp_arrow_buttons = id==14 and lppro_arrow_buttons or lpmini_arrow_buttons
	 for _, i in ipairs(lp_arrow_buttons) do
	    self:out(1, "ctl", {accent_arrows, i, ch})
	 end
	 -- initialize the launch grid from the midi map
	 self:launchpad_pads(portno)
	 -- drum grid
	 if id ~= 13 then -- not available on the Mini
	    for i = 0, 63 do
	       local color = 8*(i//16)+33
	       self:out(1, "note", {i+36, color, drumch})
	    end
	 end
	 self.launchpad_drums[portno] = false
	 if id == 14 then
	    -- LP Pro: Set up the four fader banks in advance. On the LP
	    -- Mini/X this is done on the fly, because there's only a single
	    -- fader bank on these devices.
	    for b = 0, 3 do
	       self:launchpad_fader_bank_setup(portno, b, lpmini_colors)
	    end
	 end
	 -- play/loop
	 self:launchpad_play(rolling)
	 self:launchpad_loop(self.arp.loopstate)
	 if launchpad_welcome and id ~= 14 then
	    local msg = launchpad_welcome
	    self:out(1, "sysex", {0, 32, 41, 2, id, 7, 0, 14, 0, assigned, string.byte(msg, 1, string.len(msg))})
	    lp_welcome[portno] = true
	 end
	 ::skip::
      end
   end
end

function raptor:launchpad_fini(force)
   if (force or launchpad ~= 0) and self:check_master() then
      -- iterate over all connected launchpads
      for portno, id in pairs(launchpad_id) do
	 local ch = portno==3 and 33 or portno==4 and 49
	 -- assert ch, but to be on the safe side...
	 if not ch then goto skip end
	 local drumch = ch+8
	 self:out(2, "float", {portno})
	 -- turn off all buttons
	 for i = 1, 8 do
	    -- left- and rightmost columns (the former is only on the LPPro)
	    if id == 14 then
	       self:out(1, "ctl", {0, 10*i, ch})
	    end
	    self:out(1, "ctl", {0, 10*i+9, ch})
	 end
	 if id == 12 then
	    -- Capture MIDI button on the Launchpad X
	    self:out(1, "ctl", {0, 98, ch})
	 end
	 if id == 14 then -- LPPro only
	    for i = 1, 8 do
	       -- lower bottom row
	       self:out(1, "ctl", {0, i, ch})
	       -- upper bottom row
	       self:launchpad_ccmaster(0, i, 0)
	    end
	 end
	 -- arrow buttons
	 local lp_arrow_buttons = id==14 and lppro_arrow_buttons or lpmini_arrow_buttons
	 for _, i in ipairs(lp_arrow_buttons) do
	    self:out(1, "ctl", {0, i, ch})
	 end
	 -- launch grid will be turned off automatically with the sysex
	 -- drum grid
	 if id ~= 13 then -- not available on the Mini
	    for i = 0, 63 do
	       self:out(1, "note", {i+36, 0, drumch})
	    end
	 end
	 -- switch the Launchpad back into standalone mode
	 self:out(1, "sysex", {0, 32, 41, 2, id, 16, 0})
	 ::skip::
      end
      -- wind down some shared status so that we can correctly power up again
      -- after a warm reset (fini without exiting Pd)
      launchpad_master = nil
      launchpad_last_page = {}
      rolling = 0
   end
end

function raptor:launchpad_note(atoms)
   if launchpad ~= 0 then
      local num, val, ch = table.unpack(atoms)
      local portno = ch==33 and 3 or ch==49 and 4
      if lp_welcome[portno] and launchpad_id and portno then
	 self:launchpad_welcome_off(launchpad_id[portno], portno)
	 lp_welcome[portno] = false
      end
      if portno and launchpad_trigger > 0 then
	 -- channel 1 on port #3 or #4 (launch grid)
	 local cc = num+128
	 local var
	 -- pretend that we are on port #3, to simplify MIDI mapping
	 ch = 33
	 atoms[3] = 33
	 if self.midi_learn ~= 0 then
	    -- We need to tie in with the MIDI learn system here so that we
	    -- can decide whether we want to be a toggle on the basis of the
	    -- parameter that is to be mapped, not the one that we're
	    -- currently mapped to (if any).
	    var = self.midi_learn_var
	 else
	    var = self:map_get(cc, ch)
	 end
	 local i = var and param_i[var]
	 local p = i and params[i]
	 if val >= launchpad_trigger then
	    if p and p.toggled then
	       -- toggle-like behavior, "on" is at full velocity; otherwise
	       -- send the velocity as is (momentary performance control)
	       atoms[2] = 127
	    end
	    return atoms
	 elseif val > 0 then
	    -- touch below trigger threshold; print the current mapping and
	    -- value, if any
	    if var and self:launchpad_master() then
	       local v = self.param_val[i]
	       if p and (not p.transport or not p.toggled or var == "play" or var == "click") and not p.looper then
		  if p.toggled then
		     v = v~= 0 and "on" or "off"
		  elseif p.integer or p.enum then
		     v = string.format("%d", v)
		  else
		     v = string.format("%g", v)
		  end
	       else
		  v = nil
	       end
	       if v then
		  v = string.format(" (%s)", v)
	       else
		  v = ""
	       end
	       print(string.format("%s is mapped to %s%s", self:cctostring(cc, ch), var, v))
	    end
	    return true
	 else
	    if p and not p.toggled and p.min == -p.max then
	       -- bipolar control, make it latch to the 0 position (like a
	       -- pitch bend wheel)
	       atoms[2] = 64
	    end
	    return atoms
	 end
      elseif ch == 41 or ch == 57 then
	 -- channel 9 on port #3 or port #4 (drum grid)
	 atoms[3] = 10 -- GM drum channel on port 1
	 return atoms
      end
   end
   return false
end

local launchpad_models = { [12] = "X", [13] = "Mini MK3", [14] = "Pro MK3" }

function raptor:launchpad_model_name(id)
   return launchpad_models[id] or "??"
end

-- This timer is used to detect long presses on the fader bank buttons. We
-- actually use two of these, one for each Launchpad port, along with
-- corresponding status variables, in order to prevent race conditions if two
-- separate Launchpads are connected at the same time.

function raptor:launchpad_fader_timer_on(portno)
   -- kick off the momentary timer, initial state 0 (waiting for timer)
   self.launchpad_momentary[portno] = 0
   -- threshold for momentary changes
   self.launchpad_clock[portno]:delay(500)
end

function raptor:launchpad_fader_timer_off(portno)
   if self.launchpad_momentary[portno] == 0 then
      self.launchpad_clock[portno]:unset()
      self.launchpad_momentary[portno] = nil
   end
end

function raptor:launchpad_fader_timer_cb(portno)
   -- switch to long-press state
   self.launchpad_momentary[portno] = 1
end

function raptor:launchpad_fader_timer_cb3()
   self:launchpad_fader_timer_cb(3)
end

function raptor:launchpad_fader_timer_cb4()
   self:launchpad_fader_timer_cb(4)
end

function raptor:lpmaster(id)
   if id then
      return id
   elseif time_master then
      -- fall back to the time master
      return time_master
   else
      -- fall back to self
      return self.id
   end
end

function raptor:launchpad_master()
   -- select the Raptor instance that gets to send all feedback
   if launchpad_master then
      -- already selected, move along
   else
      -- try the ccmaster, time master, and self, in that order
      launchpad_master = self:lpmaster(self.ccmaster)
   end
   -- check that we are the golden one
   return launchpad_master == self.id
end

local launchpad_last_page = {}

function raptor:launchpad_master_change(old_id, new_id)
   -- This gets invoked when the time master or ccmaster changes, in which
   -- case we may need to update the fader pages and the launch grids on all
   -- connected Launchpads accordingly.
   if not launchpad_id or not next(launchpad_id) then
      -- We need the launchpad_id table to be populated which may not be the
      -- case during startup. Later, there's nothing to do here if we don't
      -- have any Launchpads connected, so we bail out in that case as well.
      return
   end
   local old_master = self:lpmaster(old_id)
   local new_master = self:lpmaster(new_id)
   -- We only execute this in the new master, and there's nothing to do if the
   -- master didn't change.
   if new_master ~= old_master and self.id == new_master then
      --assert(not launchpad_master or old_master == launchpad_master)
      -- hand over to the new instance (i.e., self)
      launchpad_master = self.id
      --print(string.format("hand over %d -> %d", old_master, new_master))
      for portno, id in pairs(launchpad_id) do
	 self:launchpad_fader_timer_off(portno)
	 local page = launchpad_last_page[portno]
	 --print(string.format("#%d (%d), page %s", portno, id, page and string.format("%d", page) or "none"))
	 self:launchpad_fader_page(portno, page)
	 self:launchpad_pads(portno)
	 self:launchpad_loop(self.arp.loopstate)
      end
   end
end

function raptor:launchpad_fader_set_page(portno, page)
   if launchpad_id then
      if page then
	 self:launchpad_fader_bank(portno, page)
	 self.launchpad_faders[portno] = page
	 self.lp_fader_map[portno] = nil
	 launchpad_last_page[portno] = page
      else
	 self.launchpad_faders[portno] = -1
	 self.lp_fader_map[portno] = nil
	 launchpad_last_page[portno] = nil
      end
   end
end

function raptor:launchpad_fader_page(portno, page)
   if launchpad ~= 0 and self:launchpad_master() then
      --print("fader page", tostring(page), "on id", self.id)
      local id = launchpad_id[portno]
      -- assert id
      self:outlet(2, "float", {portno})
      if page then
	 -- switch to the new page
	 -- initialize the fader values via MIDI feedback
	 if id == 14 then
	    -- LP Pro: The first four fader banks are available as separate
	    -- pages on the fader layout (#1). For the fifth page, we swap out
	    -- the first fader bank on the fly.
	    if page == 0 or page == 4 then
	       self:launchpad_fader_bank_setup(portno, page, lpmini_colors)
	    end
	    self:outlet(1, "sysex", {0, 32, 41, 2, id, 0, 1, page<4 and page or 0, 0})
	 else
	    -- LP Mini/X: We need to set up the fader page here on the fly,
	    -- since there's just a single bank of these and a single fader
	    -- layout (#13).
	    self:launchpad_fader_bank_setup(portno, page, lpmini_colors)
	    self:outlet(1, "sysex", {0, 32, 41, 2, id, 0, 13})
	 end
	 -- update the internal state
	 self:launchpad_fader_set_page(portno, page)
	 -- just in case we're still waiting for the timer
	 self:launchpad_fader_timer_off(portno)
      else
	 -- Switch back to the previous non-fader page. Use note mode as
	 -- default if for some reason we never received a layout message.
	 if self.launchpad_page[portno] then
	    page = self.launchpad_page[portno]
	 else
	    page = id==12 and {1} or id==13 and {5} or {4,0}
	 end
	 if id == 14 then
	    -- the protocol demands an extra zero on the Pro, not sure why
	    table.insert(page, 0)
	 end
	 self:outlet(1, "sysex", {0, 32, 41, 2, id, 0, table.unpack(page)})
	 -- update the internal state
	 self:launchpad_fader_set_page(portno)
	 -- kill off the timer if needed
	 self:launchpad_fader_timer_off(portno)
      end
   end
end

function raptor:launchpad_ctl(atoms)
   if launchpad ~= 0 then
      local val, num, ch = table.unpack(atoms)
      local portno = ch==33 and 3 or ch==49 and 4
      if lp_welcome[portno] and launchpad_id and portno then
	 self:launchpad_welcome_off(launchpad_id[portno], portno)
	 lp_welcome[portno] = false
      end
      if portno then
	 -- channel 1 on port 3 or 4
	 local portno = ch == 33 and 3 or 4
	 local id = launchpad_id[portno]
	 if not id then
	    -- not a known device, bail out
	    return false
	 end
	 -- pretend that we are on port #3, to simplify MIDI mapping
	 atoms[3] = 33
	 local function layout_request(num)
	    local function is_layout_button(num)
	       -- The Pro doesn't have any detectable layout buttons at all, the
	       -- X three, the Mini four.
	       local first = id==14 and 99 or 95
	       local last = id==12 and 97 or 98
	       return num >= first and num <= last
	    end
	    if is_layout_button(num) then
	       if id ~= 14 and self:launchpad_master() then
		  -- On the Launchpad Mini and X, we don't get any automatic
		  -- layout change messages from the device, so we request
		  -- one; the result will be processed by launchpad_sysex.
		  self:outlet(2, "float", {portno})
		  self:outlet(1, "sysex", {0, 32, 41, 2, id, 0})
	       end
	       return true
	    else
	       return false
	    end
	 end
	 -- For the Note layout button on the Launchpad X (CC96), we do some
	 -- magic to switch between notes and drum view. For the Launchpad
	 -- Pro, we use the Clear button (CC60) instead (alas, the Note button
	 -- by itself doesn't generate any MIDI data on the Pro), and check
	 -- that we're currently in note mode.
	 local last_page = self.launchpad_page[portno] and self.launchpad_page[portno][1]
	 local current_fader_page = self.launchpad_faders[portno] or -1
	 local notes_num = id==12 and last_page == 1 and 96 or id==14 and current_fader_page<0 and last_page == 4 and 60
	 if num == notes_num then
	    if val > 0 and self:launchpad_master() then
	       -- toggles drum mode
	       self.launchpad_drums[portno] = not self.launchpad_drums[portno]
	       local flag = self.launchpad_drums[portno] and 1 or 0
	       self:outlet(2, "float", {portno})
	       if id == 14 then
		  -- Launchpad Pro MK3. This isn't what the prog manual
		  -- says; the message given there is identical to the
		  -- Launchpad X (see below), which won't work on the
		  -- Pro. This one works, though, I gleaned it from the
		  -- output of Bitwig Studio.
		  self:outlet(1, "sysex", {0, 32, 41, 2, id, 0, flag+1})
	       elseif id == 13 then
		  -- There's no drum rack on the Mini, ignore.
		  return true
	       elseif id == 12 then
		  -- Launchpad X. Works pretty much like the Pro, but uses a
		  -- different sysex message.
		  self:outlet(1, "sysex", {0, 32, 41, 2, id, 15, flag})
	       end
	       layout_request(num)
	    end
	    return true
	 end
	 local page = lppro_fader_pages[num] or lpmini_fader_pages[num]
	 if page then
	    if val > 0 then
	       -- Switch to one of the five fader banks:
	       local old_page = self.launchpad_faders[portno] or -1
	       --print("fader page, old:", tostring(old_page), "new:", tostring(page))
	       if page ~= old_page then
		  -- switch to the new page
		  self:launchpad_fader_page(portno, page)
		  -- kick off the momentary timer
		  self:launchpad_fader_timer_on(portno)
	       else
		  -- switch back to the previous non-fader page
		  self:launchpad_fader_page(portno)
	       end
	    elseif self.launchpad_momentary[portno] == 0 then
	       -- still momentary, cancel the timer
	       self:launchpad_fader_timer_off(portno)
	    elseif self.launchpad_momentary[portno] == 1 then
	       -- timer has triggered already, so we're in long-press state
	       -- where we switch back to the previous non-fader page as soon
	       -- as the button is released (which we just detected)
	       self:launchpad_fader_page(portno)
	    end
	 elseif num >= 101 and num <= 108 then
	    if val > 0 then
	       self:in_1_ccmaster_set({num-100})
	    end
	 else
	    -- arrow buttons: up, down, left, right
	    local lp_arrow_buttons = id==14 and lppro_arrow_buttons or lpmini_arrow_buttons
	    local up, down, left, right = table.unpack(lp_arrow_buttons)
	    if num == left then
	       if val > 0 then
		  self:in_1_ccmaster_prev()
	       end
	    elseif num == right then
	       if val > 0 then
		  self:in_1_ccmaster_next()
	       end
	    elseif num == up then
	       if val > 0 and self:check_ccmaster() then
		  local i = self.presetno and self.presetno or 1
		  i = i-1
		  self:recall_preset(i)
	       end
	    elseif num == down then
	       if val > 0 and self:check_ccmaster() then
		  local i = self.presetno and self.presetno or 1
		  i = i+1
		  self:recall_preset(i)
	       end
	    elseif num == 98 and id ==12 then
	       -- make the "Capture MIDI" button on the X MIDI-mappable
	       return false
	    elseif not layout_request(num) then
	       return false
	    end
	 end
	 return true
      elseif ch == 37 or ch == 53 then
	 -- output from faders
	 local val, cc, ch = table.unpack(atoms)
	 -- we need to keep track of these values to break MIDI feedback loops
	 --print("in", cc, val)
	 self.lp_fader_val[cc] = val
	 -- some controls are also on the extra page, update these as well
	 local alt_cc = self.lp_alt_cc[cc]
	 if alt_cc then
	    --print("alt_cc feedback", cc, alt_cc)
	    self.lp_fader_val[alt_cc] = val
	 end
	 -- direct sync between LP ports
	 if self:launchpad_master() then
	    -- swap ports
	    local portno = ch == 37 and 4 or 3
	    local id = launchpad_id[portno]
	    if id then
	       self:lp_out(portno, id, val, cc)
	       if alt_cc then
		  self:lp_out(portno, id, val, alt_cc)
	       end
	    end
	 end
	 -- we just pretend that these all come from port 3, to simplify the
	 -- MIDI mapping
	 atoms[3] = 37
	 return atoms
      end
   end
   return false
end

-- DEVICE INQUIRY: At startup, we first send out a device inquiry sysex on
-- port #3+4, so that we can determine the launchpad ids for the devices on
-- those ports. (If we get this wrong, you can also manually set the
-- launchpad_id table near the beginning of the source.)

function raptor:launchpad_sysex(atoms, portno)
   if launchpad ~= 0 then
      -- check whether this is an identity reply message
      if portno==3 or portno==4 then
	 local idreq = {126, 0, 6, 2, 0, 32, 41, 0, 1}
	 for i = 1, #idreq do
	    if atoms[i] ~= idreq[i] and i~=2 and i~=8 then
	       -- not for us, pass
	       goto skip
	    end
	 end
	 -- identity reply, this is the critical number:
	 local rid = atoms[8]
	 -- 14 = Launchpad Pro MK3, 13 = Launchpad Mini MK3, 12 = Launchpad X
	 local id = rid==0x23 and 14 or rid==0x13 and 13 or rid==0x03 and 12
	 if not id then
	    pd.post(string.format("WARNING: unknown Launchpad device %xh on port #%d", rid, portno))
	 elseif launchpad_id[portno] == id then
	    -- another Launchpad, same model, this can be safely ignored
	 elseif launchpad_id[portno] then
	    pd.post(string.format("WARNING: Launchpad %s conflicts with Launchpad %s on port #%d", self:launchpad_model_name(id), self:launchpad_model_name(launchpad_id[portno]), portno))
	 else
	    launchpad_id[portno] = id
	 end
      else
	 return false
      end
      ::skip::
      -- check whether this is a Launchpad message
      local lp_id = {0, 32, 41, 2}
      for i = 1, #lp_id do
	 if atoms[i] ~= lp_id[i] then
	    -- not for us, pass
	    return false
	 end
      end
      if atoms[5] < 12 or atoms[5] > 14 then
	 -- unknown model, bail out
	 return false
      end
      --print("sysex", table.unpack(atoms))
      if atoms[6] == 0 then
	 -- layout/page change, we want to record this unless it's a fader
	 -- page; this message is different on the Pro, as it has many more
	 -- layouts and different pages per layout
	 local fader_mode = atoms[5] == 14 and 1 or 13
	 if atoms[7] ~= fader_mode then
	    self:launchpad_fader_set_page(portno)
	    -- atoms[8] will only be set on the Pro
	    self.launchpad_page[portno] = {atoms[7], atoms[8]}
	    self:launchpad_fader_timer_off(portno)
	 end
      end
      return true
   else
      return false
   end
end

-- feedback

-- All actual feedback operations are to be executed in the launchpad_master
-- instance, so that we don't send out identical messages from each instance.
-- Besides launchpad_ccmaster(), which only updates its own button for the
-- ccmaster display on the LP Pro, the only exceptions are launchpad_pads()
-- and launchpad_pulse(), which are both executed in the time master -- the
-- former because it is executed during startup, and the latter because it is
-- the pulse display which should reflect the Raptor parameter settings of the
-- time master instance.

-- Turn off the startup animation (scrolling text, X and Mini only).
function raptor:launchpad_welcome_off(id, portno)
   self:out(2, "float", {portno})
   self:out(1, "sysex", {0, 32, 41, 2, id, 7})
end

-- This needs to be global data (set at LP initialization time from the MIDI
-- map) shared by all instances, so that the current launchpad_master knows
-- about which button on the Launchpad grid is bound to which function.
-- (Right now, this information is used to update toggles like "play" and
-- "loop" on the launch grid when their status changes.)
local lp_mapped

-- Pad colors, indexed by Raptor parameters. Some functions have two colors
-- assigned to them if they're used to represent toggle states.

local lppadcolor = {
   -- toggles and triggers
   ["mute"] = {11, 61},
   ["latch"] = {23, 21},
   ["bypass"] = {7, 5},
   ["loop-load"] = 33,
   ["loop-save"] = 5,
   ["loop"] = {accent_loop, accent_loop-8},
   ["click"] = {16, 17},
   ["raptor"] = {2, 3},
   ["uniq"] = {104, 79},
   ["rewind"] = 58,
   ["loop-prev"] = accent_arrows,
   ["loop-next"] = accent_arrows,
   ["play"] = {3, 25},
   -- parameters from fader bank 1
   ["pitchhi"] = lpmini_colors[1],
   ["pitchlo"] = lpmini_colors[1],
   ["mode"] = lpmini_colors[1],
   ["pos"] = lpmini_colors[1],
   ["tempo"] = lpmini_colors[1],
   ["meter-num"] = lpmini_colors[1],
   ["meter-denom"] = lpmini_colors[1],
   ["division"] = lpmini_colors[1],
   -- parameters from fader bank 2
   ["velmod"] = lpmini_colors[2],
   ["pmod"] = lpmini_colors[2],
   ["gain"] = lpmini_colors[2],
   ["gatemod"] = lpmini_colors[2],
   ["hmod"] = lpmini_colors[2],
   ["prefmod"] = lpmini_colors[2],
   ["smod"] = lpmini_colors[2],
   ["nmod"] = lpmini_colors[2],
   -- parameters from fader bank 3
   ["minvel"] = lpmini_colors[3],
   ["pmin"] = lpmini_colors[3],
   ["wmin"] = lpmini_colors[3],
   ["gate"] = lpmini_colors[3],
   ["hmin"] = lpmini_colors[3],
   ["pref"] = lpmini_colors[3],
   ["smin"] = lpmini_colors[3],
   -- uniq is handled as a toggle, see above
   --["uniq"] = lpmini_colors[3],
   -- parameters from fader bank 4
   ["maxvel"] = lpmini_colors[4],
   ["pmax"] = lpmini_colors[4],
   ["wmax"] = lpmini_colors[4],
   ["gate"] = lpmini_colors[4],
   ["hmax"] = lpmini_colors[4],
   ["pref"] = lpmini_colors[4],
   ["smax"] = lpmini_colors[4],
   ["nmax"] = lpmini_colors[4]
}

function raptor:get_lppadcolor(var, state)
   if var then
      local color = lppadcolor[var]
      if type(color) == "table" then
	 if not state then
	    -- the actual state for these toggles is *not* the one in param
	    -- storage, we need to get it elsewhere
	    local toggles = {
	       mute = self.mute, bypass = self.bypass,
	       latch = self.arp.latch and 1 or 0,
	       play = rolling, loop = self.arp.loopstate
	    }
	    local val = toggles[var]
	    if not val then
	       -- other params can be fetched straight from storage
	       local i = param_i[var]
	       if i then
		  val = self.param_val[i]
	       end
	    end
	    state = val and val ~= 0 and 1 or 0
	 end
	 color = color[state+1]
      elseif not color then
	 color = assigned
      end
      return color
   else
      return blank
   end
end

function raptor:launchpad_update_pages()
   -- Status information about the current fader bank that needs to be updated
   -- in every instance, no actual feedback is generated right here.
   if launchpad ~= 0 and launchpad_id and next(launchpad_id) then
      self.lp_fader_val = {}
      self.lp_alt_cc = {}
      for portno, id in pairs(launchpad_id) do
	 self.lp_fader_map[portno] = nil
      end
   end
end

function raptor:launchpad_iter(fun)
   if launchpad_id then
      -- iterate over all Launchpad ports
      for portno, id in pairs(launchpad_id) do
	 local ch = portno==3 and 33 or portno==4 and 49
	 -- assert ch, but to be on the safe side...
	 if ch then
	    fun(ch, portno, id)
	 end
      end
   end
end

function raptor:launchpad_pulse(w, val)
   -- we only do this on the time master, no matter what the current
   -- launchpad_master is
   if  launchpad ~= 0 and self:check_master() then
      -- w is the weight, val the velocity, n the number of beats per bar to
      -- trigger, b the total number of beats.
      local n, b = launchpad_n_pulses, self.arp.beats
      local state = w >= b-n and 1 or 0
      self:launchpad_iter(function(ch, portno)
	    pd.send(string.format("%s-launchpad", self.id), "pulse", {val*state, 99, ch, portno})
      end)
   end
end

function raptor:launchpad_ccmaster(state, i, color)
   -- This gets invoked in *every* instance, each instance only sets its own
   -- ccmaster button on or off (if any, those button are only on the LP Pro).
   if launchpad ~= 0 then
      if not i then
	 i = self:get_instance()
      end
      if i > 0 and i <= 8 then
	 self:launchpad_iter(function(ch, portno, id)
	       if id == 14 then
		  -- LP Pro only. Neither the Mini nor the X have these button
		  -- rows, and I found that at least on the Mini things go
		  -- haywire when it receives CCs in the 101-108 range.
		  self:out(1, "note", {i+100, color or accent_arrows+8*state, ch})
	       end
	 end)
      end
   end
end

function raptor:launchpad_play(state)
   -- This *must* be invoked in the time master, otherwise we get the wrong
   -- transport state.
   if launchpad ~= 0 and self:check_master() then
      -- This is shared across all instances and is also used by the launchkey
      -- driver.
      rolling = state
      self:launchpad_iter(function(ch, portno, id)
	    local num = lp_mapped["play"]
	    local color = lppadcolor["play"][state+1]
	    if num then
	       self:out(1, "note", {num, color, ch})
	    end
	    if id == 14 then
	       self:out(1, "ctl", {color, 20, ch})
	    end
      end)
   end
end

function raptor:launchpad_loop(state)
   if launchpad ~= 0 and self:launchpad_master() then
      self:launchpad_iter(function(ch, portno, id)
	    local num = lp_mapped["loop"]
	    local color = lppadcolor["loop"][state+1]
	    if num then
	       self:out(1, "note", {num, color, ch})
	    end
	    if id == 14 then
	       self:out(1, "ctl", {color, 10, ch})
	    elseif id == 12 then
	       self:out(1, "ctl", {color, 98, ch})
	    end
      end)
   end
end

function raptor:tgl_lppadcolor(var)
   return type(lppadcolor[var]) == "table"
end

function raptor:launchpad_pad(var)
   -- Generic pad feedback after param changes; this updates the toggles.
   if launchpad ~= 0 and lp_mapped and self:launchpad_master() then
      local num = lp_mapped[var]
      local tgl = self:tgl_lppadcolor(var)
      if num and tgl then
	 local color = self:get_lppadcolor(var)
	 self:launchpad_iter(function(ch)
	       self:outlet(1, "note", {num, color, ch})
	 end)
      end
   end
end

function raptor:launchpad_pads(portno)
   -- This updates all the pads (and also rebuilds the lp_mapped table).
   local ch0 = portno==3 and 33 or portno==4 and 49
   -- assert ch0, but to be on the safe side...
   if launchpad ~= 0 and ch0 and self:launchpad_master() then
      local mapped = {}
      for cc, map in pairs(self.midi_map) do
	 if cc >= 128 then
	    local num = cc-128
	    for ch, v in pairs(map) do
	       if ch == 33 and v then
		  local var = type(v) == "table" and v[1] or v
		  local color = self:get_lppadcolor(var)
		  mapped[var] = num
		  self:outlet(1, "note", {num, color, ch0})
	       end
	    end
	 end
      end
      lp_mapped = mapped
   end
end

function raptor:launchpad_mapped(cc, ch, var)
   if launchpad ~= 0 and ch == 33 and cc >= 128 and self:launchpad_master() then
      local num = cc-128
      -- update the lp_mapped table
      if var then
	 lp_mapped[var] = num
      else
	 -- need to look for any mappings of num and get rid of them
	 for var1, num1 in pairs(lp_mapped) do
	    if num1 == num then
	       lp_mapped[var1] = nil
	    end
	 end
      end
      local color = self:get_lppadcolor(var)
      self:launchpad_iter(function(ch)
	    self:outlet(1, "note", {num, color, ch})
      end)
   end
end

function raptor:lpfader_to_midi(var, val, opt)
   local i = param_i[var]
   if i and self.param_val[i] then
      local param = params[i]
      -- Use the given value if any. This is intended for out-of-band
      -- parameter feedback which is being passed directly instead of the
      -- usual raptor:param() call sequence. Otherwise we fetch the current
      -- value from the parameter storage.
      if not val then
	 val = self.param_val[i]
      end
      -- map back to MIDI; we ignore tgl (opt == true) here since we only care
      -- about the parameter config itself, but we heed the polarity
      local pol = type(opt) == "number" and opt or 1
      if param.toggled then
	 val = val>0 and 127 or 0
      else
	 local min, max = param.min, param.max
	 if var == "pos" then
	    -- this one is special, it has a nominal range of -24..24, but
	    -- we also want to clamp it to the actual number of beats
	    max = math.min(max, self.arp.beats)
	    min = -max
	 end
	 if max <= min then
	    -- this can't happen?
	    val = 0
	 elseif pol < 0 then
	    -- inverted
	    val = (max-val)/(max-min)*128
	 else
	    val = (val-min)/(max-min)*128
	 end
	 -- round down to integer
	 val = math.floor(val)
	 -- clamp to MIDI data byte
	 val = math.max(0, math.min(127, val))
      end
   end
   return val
end

-- direct feedback to Launchpad
function raptor:lp_out(portno, id, val, cc)
   -- assert(self:launchpad_master())
   --print("out", portno, cc, val)
   -- I think that there's a firmware bug on the Mini MK3, which even with the
   -- latest firmware has the channels for the fader position and color sets
   -- (5 and 6) the wrong way around.
   local ch = id==13 and 38 or 37
   if portno > 3 then
      ch = ch+16
   end
   self:outlet(1, "ctl", {val, cc, ch})
end

function raptor:launchpad_fader_val(var, val)
   -- Here we need to check whether the Launchpad itself last sent the value
   -- that we now received as feedback from the param kitchen. If the fader
   -- values match up (up to rounding discrepancies), we just quietly drop the
   -- feedback value in order to prevent a feedback loop which makes the fader
   -- operation very sluggish, on the LP Mini and X at least. (The LP Pro
   -- appears to have its own internal handler for this, as the faders seem to
   -- run fine and smooth even without this whole rigmarole.)
   function check(cc, val)
      local last_val = self.lp_fader_val[cc]
      if last_val and not self.check_pickup then
	 local ivar, ival, eps = self:from_midi(last_val, cc, 37)
	 local ovar, oval, eps = self:from_midi(val, cc, 37)
	 if ival and oval then
	    local delta = math.abs(ival-oval)
	    return delta > eps
	 end
      end
      return true
   end
   if launchpad ~= 0 and var and launchpad_id and self:launchpad_master() then
      local portno1
      for portno, id in pairs(launchpad_id) do
	 local b = self.launchpad_faders[portno]
	 if b and b >= 0 then
	    local ch = 37
	    if not self.lp_fader_map[portno] then
	       -- need to reinitialize our map for the current fader bank
	       self.lp_fader_map[portno] = {}
	       local cc0 = lp_fader_bank(b)
	       for i = 0, 7 do
		  local cc = cc0 + i
		  local var, opt = self:map_get(cc, ch)
		  if var then
		     self.lp_fader_map[portno][var] = {cc, opt}
		  end
		  if portno1 and self.lp_fader_map[portno1][var] then
		     -- We also need to record alternative CC bindings for the
		     -- page on the other port here. This is used for cross
		     -- feedback between two Launchpads connected on separate
		     -- ports, if one of these has the "extra" page of faders
		     -- selected. The extra bindings overlap with the other
		     -- pages, so that we can actually have two *different*
		     -- CCs representing the same parameter at the same time.
		     local alt_cc = self.lp_fader_map[portno1][var][1]
		     if alt_cc ~= cc then
			--print("alt_cc setup", cc, alt_cc)
			self.lp_alt_cc[cc] = alt_cc
			self.lp_alt_cc[alt_cc] = cc
		     end
		  end
	       end
	       portno1 = portno
	    end
	    if self.lp_fader_map[portno] then
	       local cc = self.lp_fader_map[portno][var]
	       if cc then
		  cc, opt = table.unpack(cc)
		  local val = self:lpfader_to_midi(var, val, opt)
		  if not val then return end -- not mapped, bail out
		  -- Compare the computed feedback value against real MIDI
		  -- data sent from the Launchpad.
		  if check(cc, val) then
		     self:lp_out(portno, id, val, cc)
		  end
	       end
	    end
	 end
      end
   end
end

function raptor:launchpad_fader_bank(portno, b)
   -- assert(self:launchpad_master())
   local id = launchpad_id[portno] -- assert id
   -- single fader bank feedback
   -- b = 0 = Volume, 1 = Pan, 2 = Send A (Send), 3 = Send B (Device), 4 = Extra
   local cc0 = lp_fader_bank(b)
   local ch = 37
   for i = 0, 7 do
      local cc = cc0 + i
      local var, opt = self:map_get(cc, ch)
      if var then
	 local val = self:lpfader_to_midi(var, nil, opt)
	 if val then
	    self:lp_out(portno, id, val, cc)
	 end
      end
   end
end

-- Launchkey (tested with LK MK3 Mini, 37, and 49)

-- default knob mode (1 == Volume)
local lkmode = 1
-- default fader mode (2 == Device)
local lkfmode = 2
-- default pad mode (2 == Session)
local lkpmode = 2

-- in drum mode this sets the offset of the drum pads (-2..3)
local lkgrid = 0

local launchkey_master = nil

local launchkey_models = { [15] = "MK3", [18] = "MK3 88" }

local function launchkey_model_name(id)
   return launchkey_models[id] or "??"
end

function raptor:launchkey_init()
   if launchkey ~= 0 and self:check_master() then
      -- switch the Launchkey into DAW/session mode
      self:out(1, "note", {12, 127, 32})
      -- set the default knob and fader mode
      self:out(1, "ctl", {lkmode, 9, 32})
      self:out(1, "ctl", {lkfmode, 10, 32})
      -- set the default pad mode
      self:out(1, "ctl", {lkpmode, 3, 32})
      -- populate the session pads
      self:launchkey_pads()
      -- initialize the drum pads
      self:launchkey_drums()
      -- light the arrow buttons (32 = 25%, I guess)
      -- NOTE: Most of the smaller buttons have no backlight on the larger LK
      -- models; on the Mini they all do. Same applies to the transport
      -- buttons below (play, loop).
      for num = 102, 107 do
	 self:out(1, "ctl", {32, num, 32})
      end
      -- play/loop
      self:launchkey_play(rolling)
      self:launchkey_loop(self.arp.loopstate)
      -- device select buttons
      for i = 1, 8 do
	 self:launchkey_ccmaster_state(0, i, 0)
      end
   end
end

local launchkey_check

function raptor:launchkey_init2()
   -- NOTE: These sysex messages *must* be sent some time after the MIDI data
   -- which puts the Launchkey into DAW mode. Otherwise they may arrive early
   -- and be ignored. (At least that's what I saw on Linux with ALSA.)
   -- Therefore we have a secondary initialization phase here which gets
   -- executed at a later time.
   if launchkey ~= 0 and self:check_master() then
      if launchkey_id then
	 if not launchkey_check then
	    -- Detected a Launchkey device during startup. Unlike the Launchpad
	    -- driver, we don't depend on this, but let's tell the user.
	    print(string.format("Launchkey %s connected on port #2", launchkey_model_name(launchkey_id)))
	    launchkey_check = true
	 end
      else
	 -- this should work in most cases, otherwise you can set the id in
	 -- the LK config section
	 launchkey_id = 15 -- LK Mini/25/37/49/61 MK3
      end
      -- initialize the display
      self:launchkey_welcome(launchkey_welcome)
      -- populate the param display
      self:launchkey_knobs()
      self:launchkey_faders()
   end
   if launchkey ~= 0 then
      -- device select buttons
      self:launchkey_ccmaster_state(0)
   end
end

function raptor:launchkey_fini(force)
   if (force or launchpad ~= 0) and self:check_master() then
      -- clear the display
      self:launchkey_welcome()
      -- session pads
      for num = 96, 103 do
	 self:out(1, "note", {num, 0, 17})
      end
      for num = 112, 119 do
	 self:out(1, "note", {num, 0, 17})
      end
      -- drum pads
      for num = 36, 51 do
	 self:out(1, "note", {num, 0, 26})
      end
      -- arrow buttons
      for num = 102, 107 do
	 self:out(1, "ctl", {0, num, 32})
      end
      -- play/loop
      self:out(1, "ctl", {0, 115, 32})
      self:out(1, "ctl", {0, 117, 32})
      -- device select buttons
      for i = 1, 8 do
	 self:launchkey_ccmaster_state(0, i, 0)
      end
      -- switch the Launchkey back to standalone mode
      self:out(1, "note", {12, 0, 32})
      -- wind down some shared status so that we can correctly power up again
      -- after a warm reset (fini without exiting Pd)
      launchkey_master = nil
      rolling = 0
   end
end

-- pseudo device select mode which also works on the Mini (uses the
-- Stop/Solo/Mute key and the bottom pad row in session mode)
local lk_select = 0

function raptor:launchkey_note(atoms)
   if launchkey ~= 0 then
      local num, val, ch = table.unpack(atoms)
      if ch == 26 then
	 -- drum pads, remap to channel 10 and transpose
	 atoms[1] = num + lkgrid*16
	 atoms[3] = 10
	 return atoms
      elseif ch == 17 and num >= 64 and num <= 71 then
	 -- device select buttons, we use these to change the ccmaster
	 if val > 0 then
	    self:in_1_ccmaster_set({num-63})
	 end
	 return true
      elseif lk_select ~= 0 and ch == 17 and num >= 112 and num <= 119 then
	 -- same for pseudo device select mode, these use the lower pad row in
	 -- session mode instead
	 if val > 0 then
	    self:in_1_ccmaster_set({num-111})
	 end
	 return true
      end
      -- everything else goes straight through to be MIDI-mapped
   end
   return false
end

-- Mapping of the Launchkey knob modes (a.k.a. Volume, Pan, Send A, Send B).
-- NOTE: Mode 2 of the Launchkey (Device) isn't used for anything special, but
-- the CCs (CC21-28) are passed straight through, so you can still map them.
local lk_knob = { [1] = 76, [2] = 20, [3] = 48, [4] = 12, [5] = 28 }

function raptor:launchkey_ctl(atoms)
   -- Kludge: We need to mess with some of the CC data for buttons only on the
   -- bigger LK models, even if the driver is off. Specifically, four of the
   -- buttons on the LK 25+ (Capture MIDI, Quantise, Click, Undo), and the
   -- nine faders on the LK 49+ partially overlap with some of our fader banks
   -- (which can't be moved for compatibility with the Launch Control XL). We
   -- move them to the CC block 57-69 on channel 32 which currently isn't used
   -- for anything else (fingers crossed), so that they can be remapped.
   -- XXXFIXME: We can't be sure what the knobs and faders are mapped to in
   -- any of the custom modes, so you'll need to make sure that these don't
   -- conflict with any of our bindings, or just don't use them with Raptor.
   local val, num, ch = table.unpack(atoms)
   if ch == 32 and num >= 74 and num <= 77 then
      atoms[2] = num-8
      return atoms
   elseif ch == 32 and num >= 53 and num <= 61 and launchkey == 0 then
      -- only remap these if the driver is inactive, otherwise the driver does
      -- its own mapping, see below
      atoms[2] = num+4
      return atoms
   end
   if launchkey ~= 0 then
      if ch == 17 or ch == 32 then
	 if num == 3 and ch == 32 then
	    -- pad mode, 1 == Drum, 2 == Session
	    val = math.floor(val)
	    if val ~= lkpmode and self:launchkey_master() then
	       if lk_select ~= 0 then
		  -- terminate pseudo device select mode if it is active
		  lk_select = 0
		  self:launchkey_pads()
	       end
	       lkpmode = val
	       -- Set the new mode on *all* connected Launchkeys. NOTE: We
	       -- only do this for modes which are also supported on the Mini,
	       -- lest a connected Mini would force us back to drum mode. The
	       -- Mini modes are restricted to Session, Drum, and Custom 1-4.
	       if lkpmode <= 2 or lkpmode >= 5 and lkpmode < 9 then
		  self:out(1, "ctl", {lkpmode, 3, 32})
	       end
	       if lkpmode == 1 then
		  -- provide feedback when entering Drum mode
		  self:launchkey_drums(true)
	       end
	    end
	 elseif num == 9 and ch == 32 then
	    -- knob mode, used to map the knobs to our usual 5 CC banks
	    val = math.floor(val)
	    if val ~= lkmode and self:launchkey_master() then
	       lkmode = val
	       -- set the new mode on *all* connected Launchkeys
	       self:out(1, "ctl", {lkmode, 9, 32})
	       self:launchkey_knobs()
	    end
	 elseif num == 10 and ch == 32 then
	    -- fader mode, used to map the faders to the 4 available CC banks
	    -- (same as the knob modes, except that mode 3 is not available)
	    val = math.floor(val)
	    if val ~= lkfmode and self:launchkey_master() then
	       lkfmode = val
	       -- set the new mode on *all* connected Launchkeys
	       self:out(1, "ctl", {lkfmode, 10, 32})
	       self:launchkey_faders()
	    end
	 -- NOTE: Buttons 51 and 52 are only available on the larger LK models
	 -- (25 and up), not on the Launchkey Mini.
	 elseif num == 51 and ch == 32 then
	    -- device select (no actual state change in Raptor, but device
	    -- select mode is active while this key is pressed, and we do
	    -- handle ccmaster selection on pads 64-71 on channel 17 and the
	    -- corresponding feedback elsewhere)
	 elseif num == 52 and ch == 32 then
	    -- device lock (this doesn't actually lock anything in Raptor, it
	    -- just redisplays the current ccmaster, or "omni" if there isn't
	    -- one, i.e., we're in omni mode)
	    if val>0 and self:launchkey_master() then
	       local flag = self.ccmaster and 1 or 0
	       --assert(flag == 0 or self.ccmaster == self.id)
	       self:launchkey_ccmaster(flag)
	    end
	 elseif not self.shift and num == 105 and ch == 17 then
	    -- pseudo device select mode which also works on the Mini (uses
	    -- the Stop/Solo/Mute key and the bottom pad row in session mode)
	    if self:launchkey_master() then
	       lk_select = val>0 and 1 or 0
	       if lk_select ~= 0 then
		  -- switch to session mode if necessary
		  if lkpmode ~= 2 then
		     lkpmode = 2
		     self:out(1, "ctl", {lkpmode, 3, 32})
		  end
		  self:launchkey_ccmaster_pads()
	       else
		  self:launchkey_pads()
	       end
	    end
	 elseif num >= 21 and num <= 28 and ch == 32 then
	    -- knobs, remapped to the 5 CC banks
	    local cc0 = lk_knob[lkmode]
	    if cc0 then
	       atoms[2] = cc0+num-20
	    end
	    return atoms
	 elseif num >= 53 and num <= 61 and ch == 32 then
	    if num <= 60 then
	       -- 8 faders, remapped to the 4 CC banks (5 CC banks of the
	       -- knobs, minus Pan mode which isn't available for the faders)
	       local cc0 = lk_knob[lkfmode]
	       if cc0 then
		  atoms[2] = cc0+num-52
	       else
		  -- default remapping, so that the CCs don't overlap with our
		  -- CC banks
		  atoms[2] = num+4
	       end
	    else
	       -- the 9th fader is special; it maps to CC7 (the volume
	       -- control) no matter what mode you're in
	       atoms[2] = 7
	    end
	    return atoms
	 elseif num >= 37 and num <= 44 and ch == 32 then
	    -- track select buttons (LK 49+), we use these to set the
	    -- ccmaster, as an alternative to device select mode which will
	    -- work on all models but the Mini (LK 25+)
	    if val > 0 then
	       self:in_1_ccmaster_set({num-36})
	    end
	 elseif num == 116 and ch == 32 then
	    -- Stop button (not on the Mini MK3): Raptor has no equivalent,
	    -- but we can simulate this function with a press of the Play
	    -- button if transport is currently rolling.
	    if rolling ~= 0 and val > 0 then
	       atoms[2] = 115
	       return atoms
	    else
	       -- ignore
	       return true
	    end
	 elseif not self.shift and
	    (num >= 102 and num <= 103 and ch == 32 or
	     num >= 106 and num <= 107 and ch == 32) then
	    -- unshifted arrow buttons
	    if val > 0 then
	       local up, down, left, right = 106, 107, 103, 102
	       if num == left then
		     self:in_1_ccmaster_prev()
	       elseif num == right then
		  self:in_1_ccmaster_next()
	       elseif lkpmode == 1 then
		  -- in drum mode, the up and down arrows shift the drumpads
		  -- in increments of 16 pads, for a total of 96 notes in six
		  -- 4x4 grids ranging from 4-19 to 84-99
		  if self:launchkey_master() then
		     if num == up and lkgrid < 3 then
			lkgrid = lkgrid+1
		     elseif num == down and lkgrid > -2 then
			lkgrid = lkgrid-1
		     end
		     self:launchkey_drums(true)
		  end
	       elseif num == up then
		  if self:check_ccmaster() then
		     local i = self.presetno or 1
		     i = i-1
		     self:recall_preset(i)
		  end
	       elseif num == down then
		  if self:check_ccmaster() then
		     local i = self.presetno or 1
		     i = i+1
		     self:recall_preset(i)
		  end
	       end
	    end
	 elseif num == 108 and ch == 17 then
	    -- shift status; this activates the arrow buttons on the Mini
	    self.shift = val > 0
	 elseif not self.shift then
	    -- the remaining bindings use the shift button
	    return false
	 elseif num == 115 and ch == 32 then
	    -- Shifted Play button: We use this to emulate the functionality
	    -- of the Stop button on the Mini MK3 which doesn't have one.
	    -- But the same will also work on the bigger Launchkeys.
	    if rolling ~= 0 and val > 0 then
	       return atoms
	    else
	       -- ignore
	       return true
	    end
	 elseif num >= 102 and num <= 103 and ch == 32 or
	    num >= 104 and num <= 105 and ch == 17 then
	    -- shifted arrow buttons (Mini MK3)
	    if val > 0 then
	       local up, down, left, right = 104, 105, 103, 102
	       if num == left then
		     self:in_1_ccmaster_prev()
	       elseif num == right then
		     self:in_1_ccmaster_next()
	       elseif lkpmode == 1 then
		  -- drum mode, up and down arrows
		  if self:launchkey_master() then
		     if num == up and lkgrid < 3 then
			lkgrid = lkgrid+1
		     elseif num == down and lkgrid > -2 then
			lkgrid = lkgrid-1
		     end
		     self:launchkey_drums(true)
		  end
	       elseif num == up then
		  if self:check_ccmaster() then
		     local i = self.presetno or 1
		     i = i-1
		     self:recall_preset(i)
		  end
	       elseif num == down then
		  if self:check_ccmaster() then
		     local i = self.presetno or 1
		     i = i+1
		     self:recall_preset(i)
		  end
	       end
	    end
	 else
	    return false
	 end
	 return true
      end
   end
   return false
end

function raptor:launchkey_sysex(atoms, portno)
   if launchkey ~= 0 then
      -- check whether this is an identity reply message
      if portno==2 then
	 local idreq = {126, 0, 6, 2, 0, 32, 41, 0, 1}
	 for i = 1, #idreq do
	    if atoms[i] ~= idreq[i] and i~=2 and i~=8 then
	       -- not for us, pass
	       return false
	    end
	 end
	 -- identity reply, this is the critical number:
	 local rid = atoms[8]
	 -- 02h = LK Mini, 34h-37h = LK 25-61, 40h = LK 88; we can all treat
	 -- these the same, except the LK 88 which has a different sysex id
	 local id = rid==0x64 and 18 or
	    (rid==0x02 or rid>=0x34 and rid<=0x37) and 15
	 if not id then
	    pd.post(string.format("WARNING: unknown Launchkey device %xh on port #%d", rid, portno))
	 elseif launchkey_id == id then
	    -- another Launchkey, similar model (same sysex id), no problem
	 elseif launchkey_id then
	    pd.post(string.format("WARNING: Launchkey %s conflicts with Launchkey %s on port #%d", launchkey_model_name(id), launchkey_model_name(launchkey_id), portno))
	 else
	    launchkey_id = id
	 end
      else
	 return false
      end
      return true
   else
      return false
   end
end

-- feedback (similar to the Launchpad, but simpler)

local lk_mapped

function raptor:launchkey_master()
   -- select the Raptor instance that gets to send all feedback
   if launchkey_master then
      -- already selected, move along
   else
      -- try the ccmaster, time master, and self, in that order
      -- we borrow self:lpmaster() from the Launchpad driver here
      launchkey_master = self:lpmaster(self.ccmaster)
   end
   -- check that we are the golden one
   return launchkey_master == self.id
end

function raptor:launchkey_master_change(old_id, new_id)
   -- This gets invoked when the time master or ccmaster changes, in which
   -- case we may need to update the launch grid accordingly.
   -- we borrow self:lpmaster() from the Launchpad driver here
   local old_master = self:lpmaster(old_id)
   local new_master = self:lpmaster(new_id)
   -- We only execute this in the new master, and there's nothing to do if the
   -- master didn't change.
   if new_master ~= old_master and self.id == new_master then
      --assert(not launchkey_master or old_master == launchkey_master)
      -- hand over to the new instance (i.e., self)
      launchkey_master = self.id
      --print(string.format("hand over %d -> %d", old_master, new_master))
      self:launchkey_pads()
      self:launchkey_knobs()
      self:launchkey_faders()
      self:launchkey_loop(self.arp.loopstate)
   end
end

function raptor:launchkey_welcome(msg)
   self:out(2, "float", {2})
   if msg then
      self:out(1, "sysex", {0, 32, 41, 2, launchkey_id, 4, 0, string.byte(msg, 1, string.len(msg))})
   else
      self:out(1, "sysex", {0, 32, 41, 2, launchkey_id, 6})
   end
end

function raptor:launchkey_play(state)
   -- This *must* be invoked in the time master, otherwise we get the wrong
   -- transport state.
   if launchkey ~= 0 and self:check_master() then
      rolling = state
      local num = lk_mapped["play"]
      if num then
	 local color = self:get_lppadcolor("play", state)
	 self:out(1, "note", {num, color, 17})
      end
      self:out(1, "ctl", {127*state, 115, 32})
   end
end

function raptor:launchkey_loop(state)
   if launchkey ~= 0 and self:launchkey_master() then
      local num = lk_mapped["loop"]
      if num then
	 local color = self:get_lppadcolor("loop", state)
	 self:out(1, "note", {num, color, 17})
      end
      self:out(1, "ctl", {127*state, 117, 32})
   end
end

function raptor:launchkey_param(offs, i, var)
   --print(string.format("LK var %d: %s", i, var))
   self:outlet(1, "sysex", {0, 32, 41, 2, launchkey_id, 7, i-1+offs, string.byte(var, 1, string.len(var))})
end

function raptor:launchkey_val(offs, i, val)
   --print(string.format("LK val %d: %s", i, val))
   self:outlet(1, "sysex", {0, 32, 41, 2, launchkey_id, 8, i-1+offs, string.byte(val, 1, string.len(val))})
end

function raptor:launchkey_padval(num, var)
   -- tooltip display for the pads
   local i = param_i[var]
   local p = i and params[i]
   local v = self.param_val[i]
   if p and (not p.transport or not p.toggled or var == "play" or var == "click") and not p.looper then
      if p.toggled then
	 v = v~= 0 and "on" or "off"
      elseif p.integer or p.enum then
	 v = string.format("%d", v)
      else
	 v = string.format("%0.2f", v)
      end
   else
      v = nil
   end
   if v then
      v = string.format("%s %s", var, v)
   else
      v = var
   end
   --print(v)
   self:outlet(1, "sysex", {0, 32, 41, 2, launchkey_id, 4, 1, string.byte(v, 1, string.len(v))})
end

function raptor:launchkey_ccmaster(state)
   -- tooltip display for ccmaster switch
   local i = self:get_instance()
   if launchkey ~= 0 and self:launchkey_master() and launchkey_id and i > 0 then
      self:outlet(2, "float", {2})
      local msg = "omni"
      if state ~= 0 then
	 msg = string.format("%d %s", i, raptor.presets[self.id])
      end
      self:outlet(1, "sysex", {0, 32, 41, 2, launchkey_id, 4, 1, string.byte(msg, 1, string.len(msg))})
   end
end

function raptor:launchkey_ccmaster_state(state, i, color)
   if launchkey ~= 0 then
      if not i then
	 i = self:get_instance()
      end
      if i > 0 and i <= 8 then
	 -- device select pads
	 self:out(1, "note", {i+63, color or accent_arrows+8*state, 17})
	 -- track select buttons (LK 49+)
	 self:out(1, "ctl", {color or accent_arrows+8*state, i+36, 17})
      end
   end
end

function raptor:launchkey_ccmaster_pads()
   if launchkey ~= 0 and self:launchkey_master() and lk_select ~= 0 then
      for i = 1, 8 do
	 local id = raptor.instances[i]
	 local state = id and id == self.ccmaster and 1 or 0
	 local color = id and accent_arrows+8*state or 0
	 self:out(1, "note", {i+111, color, 17})
      end
   end
end

function raptor:launchkey_preset(name)
   -- tooltip display for preset recall
   if launchkey ~= 0 and self:launchkey_master() and launchkey_id then
      local msg = string.format("preset %s", name)
      self:outlet(2, "float", {2})
      self:outlet(1, "sysex", {0, 32, 41, 2, launchkey_id, 4, 1, string.byte(msg, 1, string.len(msg))})
   end
end

function raptor:launchkey_ccval(cc, ch, var, val, val2)
   function make_val(offs, k, val, val2)
      if k and cc >= k+1 and cc <= k+8 then
	 local i = param_i[var]
	 if i then
	    self:outlet(2, "float", {2})
	    if int_param[i] then
	       val = string.format("%4d", val)
	       val2 = val2 and string.format("%d", val2)
	    else
	       val = string.format("%5.2f", val)
	       val2 = val2 and string.format("%0.2f", val2)
	    end
	    -- val2 ~= nil means a failed pickup check, in that case we also
	    -- indicate the target value that we need to catch up to
	    if val2 then
	       val = string.format("%s [%s]", val, val2)
	    end
	    self:launchkey_val(offs, cc-k, val)
	 end
      end
   end
   if launchkey ~= 0 and self:launchkey_master() and ch == 32 then
      -- knobs
      local k = lk_knob[lkmode]
      make_val(56, k, val, val2)
      -- faders
      local k = lk_knob[lkfmode]
      make_val(80, k, val, val2)
   end
end

function raptor:launchkey_knobs()
   if launchkey ~= 0 and self:launchkey_master() then
      local k = lk_knob[lkmode]
      if k then
	 local ch = 32
	 self:outlet(2, "float", {2})
	 for i = 1, 8 do
	    local cc = k+i
	    local var = self:map_get(cc, ch)
	    if var then
	       self:launchkey_param(56, i, var)
	    end
	 end
      end
   end
end

function raptor:launchkey_faders()
   if launchkey ~= 0 and self:launchkey_master() then
      local k = lk_knob[lkfmode]
      if k then
	 local ch = 32
	 self:outlet(2, "float", {2})
	 for i = 1, 8 do
	    local cc = k+i
	    local var = self:map_get(cc, ch)
	    if var then
	       self:launchkey_param(80, i, var)
	    end
	 end
      end
   end
end

function raptor:launchkey_pads()
   if launchkey ~= 0 and self:launchkey_master() then
      local mapped = {}
      for cc, map in pairs(self.midi_map) do
	 local num = cc-128
	 if num >= 96 and num <= 103 or num >= 112 and num <= 119 then
	    for ch, v in pairs(map) do
	       if ch == 17 and v then
		  local var = type(v) == "table" and v[1] or v
		  -- we borrow the color map from the Launchpad here
		  local color = self:get_lppadcolor(var)
		  mapped[var] = num
		  if num <= 103 or lk_select == 0 then
		     self:outlet(1, "note", {num, color, ch})
		  end
	       end
	    end
	 end
      end
      lk_mapped = mapped
   end
end

function raptor:launchkey_pad(var)
   -- Generic pad feedback after param changes; this updates the toggles and
   -- shows some tooltips for the pads on the LCD display.
   if launchkey ~= 0 and lk_mapped and self:launchkey_master() then
      local num = lk_mapped[var]
      if num then
	 -- tooltip
	 self:outlet(2, "float", {2})
	 self:launchkey_padval(num, var)
	 local tgl = self:tgl_lppadcolor(var)
	 if tgl then
	    -- we borrow the color map from the Launchpad here
	    local color = self:get_lppadcolor(var)
	    self:outlet(1, "note", {num, color, 17})
	 end
      end
   end
end

function raptor:launchkey_mapped(cc, ch, var)
   local num = cc-128
   if launchkey ~= 0 and ch == 17 and
      (num >= 96 and num <= 103 or num >= 112 and num <= 119) and
      self:launchkey_master() then
      -- update the lk_mapped table
      if var then
	 lk_mapped[var] = num
      else
	 -- need to look for any mappings of num and get rid of them
	 for var1, num1 in pairs(lk_mapped) do
	    if num1 == num then
	       lk_mapped[var1] = nil
	    end
	 end
      end
      -- we borrow the color map from the Launchpad here
      local color = self:get_lppadcolor(var)
      self:outlet(1, "note", {num, color, ch})
   end
end

function raptor:launchkey_drums(show)
   -- populate the drum pads, these change color according to the subset of 16
   -- pads set with lkgrid
   for num = 36, 51 do
      local i = num-36 -- pad number 0-15
      local j = i+lkgrid*16 -- add scroll (grid -2 up to 3)
      -- The first term below is basically the same color spec as in the
      -- Launchpad drum grid which represents the six 4x4 subgrids we have on
      -- tap (the four grids of the Launchpad, plus two extra 4x4 grids below
      -- the range on the Launchpad). The second term adds an accent (darker
      -- variant of the same color) to the second group of 2x4 in the
      -- Launchkey's default drum grid layout, which would be above the first
      -- 2x4 on the Launchpad. The color coding hopefully makes it easier to
      -- find your way on the grid.
      local color = j//16*8+33 + i//8%2*2
      self:out(1, "note", {num, color, 26})
   end
   if show and self:launchkey_master() and launchkey_id then
      local i = 36+lkgrid*16
      local msg = string.format("drums %d: %d-%d", lkgrid, i, i+15)
      self:outlet(2, "float", {2})
      self:outlet(1, "sysex", {0, 32, 41, 2, launchkey_id, 4, 1, string.byte(msg, 1, string.len(msg))})
   end
end

-- APC mini mk2

-- track buttons
local apc_button = {0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b}
-- faders (mode 5 not implemented yet)
local apc_fader = { [1] = 76, [2] = 48, [3] = 12, [4] = 28, [5] = 20 }
-- fader mode 1-4
local apc_mode = 1
-- pad mode (0 == session, 1 == keys, 2 == drums)
local apc_pmode = 0

local apcmini_master = nil

function raptor:apcmini_init()
   if apcmini ~= 0 and self:check_master() then
      -- switch to the default pad mode
      self:outlet(2, "float", {apcmini_portno})
      self:outlet(1, "sysex", {0x47, 0x7f, 0x4f, 0x62, 0x00, 0x01, apc_pmode})
      local ch = (apcmini_portno-1)*16+1
      -- track buttons
      self:apcmini_mode()
      -- pads
      self:apcmini_pads()
      -- play/loop
      self:apcmini_play(rolling)
      self:apcmini_loop(self.arp.loopstate)
   end
end

function raptor:apcmini_fini(force)
   if (force or apcmini ~= 0) and self:check_master() then
      local ch = (apcmini_portno-1)*16+1
      -- track buttons
      self:apcmini_mode(0)
      -- pads
      self:apcmini_pads(0)
      -- wind down some shared status so that we can correctly power up again
      -- after a warm reset (fini without exiting Pd)
      apcmini_master = nil
      rolling = 0
   end
end

function raptor:apcmini_note(atoms)
   if apcmini ~= 0 then
      local ch = (apcmini_portno-1)*16
      local ch1, ch7, ch10 = ch+1, ch+7, ch+10
      local num, val, ch = table.unpack(atoms)
      if ch == ch10 and num >= 64 then
	 -- drum pads, rearrange as four 4x4 sections
	 num = num-64
	 local r, c = num//8, num%8
	 local i = (c>=4 and 32 or 0) + r*4 + c%4
	 atoms[1] = i+36
	 atoms[3] = 10
	 return atoms
      elseif ch == ch1 and num == 0x7a then
	 -- shift status; this enables the instance selection mode
	 self.shift = val > 0
	 -- update the buttons
	 if self.shift then
	    self:apcmini_mode(0)
	    self:apcmini_ccmaster_update()
	 else
	    self:apcmini_mode()
	 end
      elseif ch == ch1 and self.shift and num >= 0x64 and num <= 0x6b then
	 -- instance selection mode
	 if val>0 then
	    self.apcmini_ccmaster_wait = true
	    self:in_1_ccmaster_set({num-0x63})
	 end
      elseif ch == ch1 and num >= 0x64 and num <= 0x67 then
	 -- fader modes
	 if val>0 and self:apcmini_master() then
	    local mode = math.floor(num-0x63)
	    if mode ~= apc_mode then
	       apc_mode = mode
	       self:apcmini_mode()
	    end
	 end
      elseif ch == ch1 and num >= 0x68 and num <= 0x69 then
	 -- up/down arrow buttons (preset selection)
	 if val>0 and self:check_ccmaster() then
	    if num == 0x68 then
	       local i = self.presetno or 1
	       i = i-1
	       self:recall_preset(i)
	    else
	       local i = self.presetno or 1
	       i = i+1
	       self:recall_preset(i)
	    end
	 end
      elseif ch == ch1 and num >= 0x6a and num <= 0x6b then
	 -- left/right arrow buttons (ccmaster selection)
	 if val>0 then
	    self.apcmini_ccmaster_wait = true
	    if num == 0x6a then
	       self:in_1_ccmaster_prev()
	    else
	       self:in_1_ccmaster_next()
	    end
	 end
      elseif ch == ch1 then
	 -- Everything else, including the session grid, should be on channel
	 -- 1 on the APC mini port. We remap these to channel 39 to facilitate
	 -- MIDI mapping and to prevent conflicts with the Launchpad.
	 atoms[3] = 39
	 return atoms
      else
	 return false
      end
      return true
   end
   return false
end

function raptor:apcmini_ctl(atoms)
   if apcmini ~= 0 then
      local ch = (apcmini_portno-1)*16
      local ch1, ch7 = ch+1, ch+7
      local val, num, ch = table.unpack(atoms)
      if num >= 48 and num <= 56 and ch == ch1 then
	 if num <= 55 then
	    -- 8 faders, remapped to the 4 CC banks
	    local cc0 = apc_fader[apc_mode]
	    if cc0 then
	       atoms[2] = cc0+num-47
	    end
	 else
	    -- the 9th fader is special; it maps to CC7 (the volume control)
	    -- no matter what mode we're in
	    atoms[2] = 7
	 end
	 -- change MIDI channel to prevent conflicts with Launchpad
	 atoms[3] = 39
	 return atoms
      end
   end
   return false
end

function raptor:apcmini_sysex(atoms, portno)
   if apcmini ~= 0 then
      -- check whether this is an identity reply message
      if portno==3 or portno==4 then
	 local function check_idreq()
	    local idreq = {126, 0, 6, 2, 0x47, 0, 0}
	    for i = 1, #idreq do
	       if atoms[i] ~= idreq[i] and i~=2 and i~=6 then
		  -- not for us, pass
		  return false
	       end
	    end
	    if not apcmini_portno then
	       -- identity reply, this is the critical number:
	       local rid = atoms[6]
	       -- 0x4f is the APC mini mk2, 0x28 the mk1, we require the mk2
	       if rid == 0x4f then
		  apcmini_portno = portno
	       elseif rid == 0x28 then
		  print("Sorry, APC mini mk1 not supported!")
	       end
	    end
	    return true
	 end
	 local function check_mode_change()
	    local idreq = {0x47, 0x7f, 0x4f, 0x62, 0x00, 0x01, 0}
	    for i = 1, #idreq do
	       if atoms[i] ~= idreq[i] and i~=2 and i~=7 then
		  -- not for us, pass
		  return false
	       end
	    end
	    if self:apcmini_master() then
	       -- mode change, this is the value we're interested in:
	       local mode = atoms[7]
	       if mode >= 0 and mode <= 2 and mode ~= apc_pmode then
		  apc_pmode = mode
		  self:apcmini_pads()
	       end
	    end
	    return true
	 end
	 return check_idreq() or check_mode_change()
      else
	 return false
      end
      return true
   else
      return false
   end
end

-- feedback

local apc_mapped

function raptor:apcmini_master()
   -- select the Raptor instance that gets to send all feedback
   if apcmini_master then
      -- already selected, move along
   else
      -- try the ccmaster, time master, and self, in that order
      -- we borrow self:lpmaster() from the Launchpad driver here
      apcmini_master = self:lpmaster(self.ccmaster)
   end
   -- check that we are the golden one
   return apcmini_master == self.id
end

function raptor:apcmini_master_change(old_id, new_id)
   -- This gets invoked when the time master or ccmaster changes, in which
   -- case we may need to update the launch grid accordingly.
   -- we borrow self:lpmaster() from the Launchpad driver here
   local old_master = self:lpmaster(old_id)
   local new_master = self:lpmaster(new_id)
   -- We only execute this in the new master, and there's nothing to do if the
   -- master didn't change.
   if new_master ~= old_master and self.id == new_master then
      --assert(not apcmini_master or old_master == apcmini_master)
      -- hand over to the new instance (i.e., self)
      apcmini_master = self.id
      --print(string.format("hand over %d -> %d", old_master, new_master))
      self:apcmini_pads()
      self:apcmini_loop(self.arp.loopstate)
   end
end

function raptor:apcmini_play(state)
   -- This *must* be invoked in the time master, otherwise we get the wrong
   -- transport state.
   if apcmini ~= 0 and self:check_master() then
      rolling = state
      local ch = (apcmini_portno-1)*16+7
      local num = apc_mapped["play"]
      if num then
	 local color = self:get_lppadcolor("play", state)
	 self:out(1, "note", {num, color, ch})
      end
   end
end

function raptor:apcmini_loop(state)
   if apcmini ~= 0 and self:apcmini_master() then
      local ch = (apcmini_portno-1)*16+7
      local num = apc_mapped["loop"]
      if num then
	 local color = self:get_lppadcolor("loop", state)
	 self:out(1, "note", {num, color, ch})
      end
   end
end

function raptor:apcmini_mode(color)
   if apcmini ~= 0 and self:apcmini_master() then
      local ch = (apcmini_portno-1)*16+1
      if color then
	 if self:apcmini_master() then
	    for i = 1, 8 do
	       self:out(1, "note", {apc_button[i], color, ch})
	    end
	 end
      else
	 for i = 1, 4 do
	    self:out(1, "note", {apc_button[i], i==apc_mode and 1 or 0, ch})
	 end
	 for i = 5, 8 do
	    self:out(1, "note", {apc_button[i], 1, ch})
	 end
      end
   end
end

function raptor:apcmini_pads(color)
   if apcmini ~= 0 and self:apcmini_master() then
      local ch = (apcmini_portno-1)*16
      local ch7, ch10 = ch+7, ch+10
      if apc_pmode == 0 then
	 if color then
	    for num = 0, 63 do
	       self:out(1, "note", {num, color, ch7})
	    end
	 else
	    local mapped = {}
	    for cc, map in pairs(self.midi_map) do
	       local num = cc-128
	       if num >= 0 and num <= 63 then
		  for ch, v in pairs(map) do
		     if ch == 39 and v then
			local var = type(v) == "table" and v[1] or v
			-- we borrow the color map from the Launchpad here
			local color = self:get_lppadcolor(var)
			mapped[var] = num
			self:outlet(1, "note", {num, color, ch7})
		     end
		  end
	       end
	    end
	    apc_mapped = mapped
	 end
      elseif apc_pmode == 2 then
	 for num = 0, 63 do
	    local r, c = num//8, num%8
	    local i = (c>=4 and 32 or 0) + r*4 + c%4
	    local col = 8*(i//16)+33
	    self:out(1, "note", {num+64, color or col, ch10})
	 end
      end
   end
end

function raptor:apcmini_pad(var)
   -- Generic pad feedback after param changes; this updates the toggles.
   if apcmini ~= 0 and apc_mapped and self:apcmini_master() then
      local num = apc_mapped[var]
      if num then
	 local tgl = self:tgl_lppadcolor(var)
	 if tgl then
	    local ch = (apcmini_portno-1)*16+7
	    -- we borrow the color map from the Launchpad here
	    local color = self:get_lppadcolor(var)
	    self:outlet(1, "note", {num, color, ch})
	 end
      end
   end
end

function raptor:apcmini_mapped(cc, ch, var)
   local num = cc-128
   if apcmini ~= 0 and ch == 39 and num >= 0 and num <= 63 and
      self:apcmini_master() then
      -- update the apc_mapped table
      if var then
	 apc_mapped[var] = num
      else
	 -- need to look for any mappings of num and get rid of them
	 for var1, num1 in pairs(apc_mapped) do
	    if num1 == num then
	       apc_mapped[var1] = nil
	    end
	 end
      end
      -- we borrow the color map from the Launchpad here
      local color = self:get_lppadcolor(var)
      local ch = (apcmini_portno-1)*16+7
      self:outlet(1, "note", {num, color, ch})
   end
end

function raptor:apcmini_ccmaster(state)
   if apcmini ~= 0 then
      local i = self:get_instance()
      if i > 0 and i <= 8 then
	 self.apcmini_ccmaster_state = {i, state}
      else
	 self.apcmini_ccmaster_state = nil
      end
      if self.apcmini_ccmaster_wait then
	 -- Update pending, do it now. NOTE: The button updates need to be
	 -- deferred until the new ccmaster state is actually available.
	 -- That's because the ccmaster update runs through Pd's messaging
	 -- system, which isn't instantaneous.
	 self:apcmini_ccmaster_update()
	 self.apcmini_ccmaster_wait = false
      end
   end
end

function raptor:apcmini_ccmaster_update()
   if apcmini ~= 0 then
      if self.shift then
	 local ch = (apcmini_portno-1)*16+1
	 if self.apcmini_ccmaster_state then
	    local i, state = table.unpack(self.apcmini_ccmaster_state)
	    local num = apc_button[i]
	    self:outlet(1, "note", {num, state, ch})
	 end
      end
   end
end

-- Launch Control XL

-- This assumes factory preset #1 on MIDI channel 9. It uses the device hold
-- button as a shift button, and binds the device select and bank buttons to
-- ccmaster_next, ccmaster_prev, and ccmaster_set. Also includes feedback for
-- ccmaster changes on the 1-8 button row.

function raptor:launchcontrol_note(atoms)
   local num, val, ch = table.unpack(atoms)
   if ch == 25 then -- channel 9 on second input port
      if num == 105 then
	 -- device hold status
	 self.shift = val > 0
	 -- update the buttons NOW
	 self:launchcontrol_ccmaster_update()
      elseif not self.shift then
	 return false
      elseif val == 0 then
	 -- no-op
      elseif num > 72 and num <= 76 and self.shift then
	 -- 73-76 = buttons 1-4
	 -- update the buttons LATER (this needs to be deferred, see
	 -- launchcontrol_ccmaster below for details)
	 self.launchcontrol_ccmaster_wait = true
	 self:in_1_ccmaster_set({num-72})
      elseif num > 88 and num <= 92 and self.shift then
	 -- 89-92 = buttons 5-8
	 -- update the buttons LATER
	 self.launchcontrol_ccmaster_wait = true
	 self:in_1_ccmaster_set({num-84})
      end
      return true
   end
   return false
end

function raptor:launchcontrol_ctl(atoms)
   local val, num, ch = table.unpack(atoms)
   if ch == 25 then
      if val > 0 and self.shift then
	 local id = self.id
	 -- 106, 107 = left, right (ccmaster select)
	 if num == 106 then
	    -- update the buttons LATER
	    self.launchcontrol_ccmaster_wait = true
	    self:in_1_ccmaster_prev()
	 elseif num == 107 then
	    -- update the buttons LATER
	    self.launchcontrol_ccmaster_wait = true
	    self:in_1_ccmaster_next()
	 elseif num == 104 and self:check_ccmaster() then
	    local i = self.presetno and self.presetno or 1
	    i = i-1
	    self:recall_preset(i)
	 elseif num == 105 and self:check_ccmaster() then
	    local i = self.presetno and self.presetno or 1
	    i = i+1
	    self:recall_preset(i)
	 end
      elseif not self.shift then
	 return false
      end
      return true
   end
   return false
end

-- ccmaster feedback

function raptor:launchcontrol_ccmaster(state)
   if launchcontrol ~= 0 then
      local i = self:get_instance()
      if i > 0 and i <= 8 then
	 self.launchcontrol_ccmaster_state = {i, state}
      else
	 self.launchcontrol_ccmaster_state = nil
      end
      if self.launchcontrol_ccmaster_wait then
	 -- Update pending, do it now. NOTE: The button updates need to be
	 -- deferred until the new ccmaster state is actually available.
	 -- That's because the ccmaster update runs through Pd's messaging
	 -- system, which isn't instantaneous.
	 self:launchcontrol_ccmaster_update()
	 self.launchcontrol_ccmaster_wait = false
      end
   end
end

function raptor:launchcontrol_ccmaster_update()
   if launchcontrol ~= 0 then
      if self.launchcontrol_ccmaster_state then
	 local i, state = table.unpack(self.launchcontrol_ccmaster_state)
	 local num = i>4 and 84+i-4 or 72+i
	 local val = self.shift and 1 or 0
	 self:outlet(1, "note", {num, 48*val*state, 25})
      end
   end
end

-- AKAI Professional MIDIMIX

function raptor:midimix_note(atoms)
   local num, val, ch = table.unpack(atoms)
   if ch == 17 then -- channel 1 on second input port
      if num == 27 then
	 -- SOLO status (used as a shift key)
	 self.shift = val > 0
	 -- update the buttons NOW
	 self:midimix_ccmaster_update()
      elseif not self.shift then
	 return false
      elseif num == 25 or num == 26 or num <= 24 and num % 3 == 0 then
	 -- All the other bindings use shifted buttons,
	 -- 25 = BANK LEFT, 26 = BANK RIGHT, the other numbers denote the
	 -- buttons 1-8 in the bottom row.
	 if val == 0 then
	    -- no-op
	 elseif num <= 24 then
	    self.midimix_ccmaster_wait = true
	    self:in_1_ccmaster_set({num // 3})
	 elseif num == 25 then
	    self.midimix_ccmaster_wait = true
	    self:in_1_ccmaster_prev()
	 elseif num == 26 then
	    self.midimix_ccmaster_wait = true
	    self:in_1_ccmaster_next()
	 end
      else
	 -- Make sure to leave all other shifted buttons unbound so that they
	 -- can be used with MIDI learn.
	 return false
      end
      return true
   end
   return false
end

function raptor:midimix_ctl(atoms)
   -- pass
   return false
end

-- ccmaster feedback

function raptor:midimix_ccmaster(state)
   if midimix ~= 0 then
      local i = self:get_instance()
      if i > 0 and i <= 8 then
	 self.midimix_ccmaster_state = {i, state}
      else
	 self.midimix_ccmaster_state = nil
      end
      if self.midimix_ccmaster_wait then
	 -- Update pending, do it now. NOTE: The button updates need to be
	 -- deferred until the new ccmaster state is actually available.
	 -- That's because the ccmaster update runs through Pd's messaging
	 -- system, which isn't instantaneous.
	 self:midimix_ccmaster_update()
	 self.midimix_ccmaster_wait = false
      end
   end
end

function raptor:midimix_ccmaster_update()
   if midimix ~= 0 then
      if self.midimix_ccmaster_state then
	 local i, state = table.unpack(self.midimix_ccmaster_state)
	 local num = i*3
	 local val = self.shift and 1 or 0
	 -- I *think* that we should be safe here. The DJ Control has some
	 -- global controls on channel 1, but none of the numbers we use here.
	 -- Conversely, the DJ Control sends out pulse messages for its
	 -- rhythmn display as note 5 on channel 1. That's the second button
	 -- on the SOLO row for us, so you should be able to watch a rough
	 -- facsimile of the blinkenlights on that button if you press SOLO.
	 -- Give it a try. ;-)
	 self:outlet(1, "note", {num, 127*val*state, 17})
      end
   end
end

-- Nektar PACER

function raptor:pacer_note(atoms)
   -- pass
   return false
end

function raptor:pacer_ctl(atoms)
   local val, num, ch = table.unpack(atoms)
   if ch == 17 then
      if val > 0 then
	 local id = self.id
	 -- 64, 65 (Stomp 1+2) = prev, next ccmaster
	 -- 66, 67 (Stomp 3+4) = prev, next preset
	 if num == 64 then
	    self:in_1_ccmaster_prev()
	 elseif num == 65 then
	    self:in_1_ccmaster_next()
	 elseif num == 66 then
	    if self:check_ccmaster() then
	       local i = self.presetno and self.presetno or 1
	       i = i-1
	       self:recall_preset(i)
	    end
	 elseif num == 67 then
	    if self:check_ccmaster() then
	       local i = self.presetno and self.presetno or 1
	       i = i+1
	       self:recall_preset(i)
	    end
	 else
	    return false
	 end
      end
      return true
   end
   return false
end

-- Hercules DJControl (experimental)

function raptor:djcontrol_init()
   -- initialize the status variables for the DJ Control
   if djcontrol ~= 0 and not self.djdata then
      -- we maintain separate status variables for each deck
      -- TODO: only two decks supported at this time, but this should
      -- hopefully do for the Hercules controllers at least
      self.djdata = { last_delta = {0, 0}, last_count = {0, 0},
		      vinyl = {1, 1}, pos = {0, 0},
		      vol = {127, 127}, xfade = 0.5 }
   end
end

-- NOTE: I don't know about other DJ controllers, but the Hercules DJControl
-- uses different MIDI channels for global controls and the two decks, also
-- depending on the status of the shift keys: 1(4) = global (shifted), 2(5) =
-- left deck (shifted), 3(6) = right deck (shifted). Also, the pads are on
-- separate channels, 7 for the left and 8 for the right deck (no dependence
-- on shift status there, instead the shift key changes the note numbers of
-- the pads).

local function djcontrol_deck(ch)
   ch = ch-16 -- all on 2nd port
   if not ch or ch <= 0 or ch > 8 then
      -- nothing to see here, move along
      return nil
   elseif ch <= 6 then
      -- 0, 1, 2 indicates global, left, right; followed by the shift status
      return (ch-1) % 3, ch >= 4
   else
      -- pads (no shift status, report pad status instead)
      return ch-6, false, true
   end
end

-- feedback: play, cue, sync, vinyl, loop, and mute buttons, and the big
-- encoder backlight which flashes along with the rhythm

local djcontrol_button = {
   ccmaster = { num = 16, ch = {23, 24}, default = 0, on = 127 },
   big12 = { num = 48, ch = {18, 19}, default = 0, on = 127 },
   sync = { num = 5, ch = {18, 19}, default = 1, on = 127 },
   cue = { num = 6, ch = {18, 19}, default = 1, on = 127 },
   play = { num = 7, ch = {18, 19}, default = 0, on = 127 },
   mute = { num = 12, ch = {18, 19}, default = 0, on = 127 },
   vinyl = { num = 3, ch = {18, 19}, default = 1, on = 127 },
   loop = { num = 3, ch = {21, 22}, default = 0, on = 127 },
   loop_in = { num = 9, ch = {18, 19}, default = 0, on = 127 },
   loop_out = { num = 10, ch = {18, 19}, default = 0, on = 127 },
   pulse = { num = 5, ch = 17, default = 0 },
}

-- initialize and finalize

function raptor:djcontrol_state_init()
   -- change all buttons to their defaults
   if djcontrol ~= 0 and self:check_master() then
      for k, b in pairs(djcontrol_button) do
	 local state = (b.on and b.on or 127)*b.default
	 -- The ccmaster button is actually a whole range of pads.
	 -- This assumes that the device has at most 8 pads.
	 local num_buttons = k == "ccmaster" and 8 or 1
	 for offs = 0, num_buttons-1 do
	    if type(b.ch) == "table" then
	       pd.send(string.format("%s-djcontrol", self.id), "note", {b.num+offs, state, b.ch[1]})
	       pd.send(string.format("%s-djcontrol", self.id), "note", {b.num+offs, state, b.ch[2]})
	    else
	       pd.send(string.format("%s-djcontrol", self.id), "note", {b.num+offs, state, b.ch})
	    end
	 end
      end
   end
end

function raptor:djcontrol_state_fini()
   -- turn all buttons off
   if djcontrol ~= 0 and self:check_master() then
      for k, b in pairs(djcontrol_button) do
	 local num_buttons = k == "ccmaster" and 8 or 1
	 for offs = 0, num_buttons-1 do
	    if type(b.ch) == "table" then
	       self:out(1, "note", {b.num+offs, 0, b.ch[1]})
	       self:out(1, "note", {b.num+offs, 0, b.ch[2]})
	    else
	       self:out(1, "note", {b.num+offs, 0, b.ch})
	    end
	 end
      end
   end
end

-- state updates

function raptor:djcontrol_state(button, state, deck, offs)
   if not self.id then
      -- This is very early on when the id hasn't been set yet, but we need it
      -- in order to send messages to the djcontrol subpatch. Just bail out at
      -- this point, there's not much else that we can do...
      return
   end
   if djcontrol ~= 0 then
      offs = offs or 0
      if deck then
	 if deck == 0 then
	    -- do both deck 1 and 2
	    self:djcontrol_state(button, state, 1, offs)
	    self:djcontrol_state(button, state, 2, offs)
	 elseif deck > 0 and offs < 8 then
	    local b = djcontrol_button[button]
	    pd.send(string.format("%s-djcontrol", self.id), "note", {b.num+offs, state*b.on, b.ch[deck]})
	 end
      elseif self:check_master() then
	 -- global controls (backlights)
	 local b = djcontrol_button[button]
	 if b.on then
	    pd.send(string.format("%s-djcontrol", self.id), "note", {b.num+offs, state*b.on, b.ch})
	 else
	    -- pulse
	    pd.send(string.format("%s-djcontrol", self.id), "pulse", {b.num+offs, state, b.ch})
	 end
      end
   end
end

function raptor:djcontrol_ccmaster(state, i, deck)
   if not i then
      i = self:get_instance()
      deck = self.deck
   end
   if i > 0 then
      i = next(raptor.decks) == nil and i or self:locate_i_deck(i, deck)
      if i then
	 self:djcontrol_state("ccmaster", state, deck or 0, i-1)
      end
   end
end

function raptor:djcontrol_deck(state)
   self:djcontrol_state("big12", state, self.deck)
end

function raptor:djcontrol_play(state)
   self:djcontrol_state("play", state, self.deck)
end

function raptor:djcontrol_loop(state)
   self:djcontrol_state("loop", state, self.deck)
   self:djcontrol_state("loop_in", state, self.deck)
   self:djcontrol_state("loop_out", state, self.deck)
end

function raptor:djcontrol_vinyl(state, deck)
   self:djcontrol_state("vinyl", state, self.deck)
end

function raptor:djcontrol_mute(state)
   self:djcontrol_state("mute", state, self.deck)
end

function raptor:djcontrol_pulse(w, val)
   -- w is the weight, val the velocity, n the number of beats per bar to
   -- trigger, b the total number of beats.
   local n, b = djcontrol_n_pulses, self.arp.beats
   if w >= b-n then
      self:djcontrol_state("pulse", val)
   end
end

function raptor:djcontrol_note(atoms)
   local num, val, ch = table.unpack(atoms)
   local deck, shift, pads = djcontrol_deck(ch)
   if not deck then
      -- pass through
      return false
   elseif deck == 0 then
      -- Kludge: This is a global control, so the actual channel must be 1 or
      -- 4 (+16), where the latter just determines the shift status. Make sure
      -- that we reset the channel to 1 (+16) so that any MIDI mapping to the
      -- button will work as intended.
      atoms[#atoms] = 17
      -- pass through
      return false
   elseif pads then
      -- the pads are on a separate plane, must be checked first since note
      -- numbers partially overlap with the non-pad buttons
      local check =  self.deck == 0 or deck == self.deck
      if num >= 0 and num < 8 then
	 -- unshifted pads in mode 1 (labeled "HOT CUE")
	 -- cue to bar in a loop, smooth transition
	 if check then
	    if self.arp.loopstate == 0 then
	       -- we only bind this control if a loop is currently playing
	       goto skip
	    elseif val > 0 then
	       -- effective loop size
	       local l = math.min(#self.arp.loop, self.arp.loopsize)
	       if l > 0 then
		  -- beginning of the bar
		  local x = (num * self.arp.beats) % l
		  -- current position in the bar
		  local i = self.arp.idx
		  -- set the loop index
		  self.arp:set_loopidx(x + i)
	       end
	    end
	 end
	 return true
      elseif num >= 8 and num < 16 then
	 -- shifted pads in mode 1 (labeled "HOT CUE")
	 -- cue to bar in a loop, immediate
	 if check then
	    if self.arp.loopstate == 0 then
	       -- we only bind this control if a loop is currently playing
	       goto skip
	    elseif val > 0 then
	       -- effective loop size
	       local l = math.min(#self.arp.loop, self.arp.loopsize)
	       if l > 0 then
		  -- beginning of the bar
		  local x = (num * self.arp.beats) % l
		  -- set the loop index to the beginning of the bar
		  self.arp:set_loopidx(x)
	       end
	    end
	 end
	 return true
      elseif num >= 16 and num < 24 then
	 -- unshifted pads in mode 2 (labeled "STEMS" on the MK2), we use these
	 -- to do ccmaster switches
	 if val > 0 then
	    -- This is a bit tricky, since this request is going to be
	    -- processed in a single random instance which might not even have
	    -- a deck assigned to it. Thus checking self.deck isn't going to
	    -- do us any good here. Instead, we check if the global decks
	    -- table is empty, in which case we do a regular instance switch,
	    -- otherwise we try to locate an instance for the deck indicated
	    -- by the message. This should do the right thing in most
	    -- cases. But note that if you're running an ensemble where some
	    -- raptors have a deck assigned to them, while others have not,
	    -- then djcontrol won't give you access to all those
	    -- instances. (As a remedy, you can still use a secondary
	    -- controller like the MIDIMIX for that purpose.)
	    local i = next(raptor.decks) == nil and num-15 or
	       self:locate_deck_i(num-15, deck)
	    self:in_1_ccmaster_set({i})
	 end
	 return true
      end
   elseif num == 8 then
      -- jog wheel touches, reset status
      self.djdata.last_delta[deck] = 0
      self.djdata.pos[deck] = self.arp.loopidx
      if self.deck == 0 or deck == self.deck then
	 if self.transport == 0 or self.djdata.vinyl[deck] ~= 0 or self.stopped and val == 0 then
	    self.stopped = val > 0
	 end
	 local vinyl = self.transport == 0 or self.djdata.vinyl[deck] ~= 0
	 self:djcontrol_deck(val > 0 and vinyl and 1 or 0)
      end
      return true
   elseif num == 3 and not shift then
      -- VINYL button: toggle scratch mode
      if val > 0 and (self.deck == 0 or deck == self.deck) then
	 self.djdata.vinyl[deck] = self.djdata.vinyl[deck] == 0 and 1 or 0
	 -- feedback
	 self:djcontrol_vinyl(self.djdata.vinyl[deck], deck)
      end
      return true
   elseif num == 5 then
      -- SYNC button: sync playback position to the time master
      if val > 0 and self:check_master() then
	 -- we're the time master, tell all instances about our playback
	 -- position so that they can sync up to us
	 local playing = self.transport ~= 0 and not shift
	 local pos = playing and self.arp.idx or 0
	 local loop_pos = (playing and self.arp.loopstate ~= 0) and self.arp.loopidx or (not playing and 0 or nil)
	 pd.send("all-arp", "sync", {pos, loop_pos})
      end
      return true
   elseif num == 6 then
      -- CUE button: rewind to the anacrusis (pos)
      if val > 0 and (self.deck == 0 or deck == self.deck) then
	 if shift then
	    -- set the anacrusis
	    self:set_pos(self.arp.idx)
	 elseif self.transport ~= 0 and self.arp.loopstate ~= 0 then
	    -- set the loop position
	    local p = self.pos % self.arp.beats
	    self.arp:set_loopidx(p)
	    self.djdata.pos[deck] = p
	 end
	 self:do_rewind(self.pos)
      end
      return true
   end
   ::skip::
   -- skip ccmaster check if already filtered by deck
   self.assert_master = deck == self.deck
   -- filter out anything that's for the other deck, or everything if
   -- self.deck < 0 (off), or nothing if self.deck == 0 (on/omni)
   -- NOTE: confusingly, the conditions are reversed here because true
   -- means filter out, false means pass through
   return self.deck ~= 0 and deck ~= self.deck
end

function raptor:djcontrol_ctl(atoms)
   local val, num, ch = table.unpack(atoms)
   local deck, shift = djcontrol_deck(ch)
   local function xfade(x)
      -- simplistic linear crossfade, maybe we should do something more
      -- sophisticated in the future
      return {x<=0.5 and 1 or 2*(1-x), x>=0.5 and 1 or 2*x}
   end
   if not deck then
      -- pass through
      return false
   elseif deck == 0 then
      -- global control
      if num == 1 and not shift then
	 -- BROWSER
	 if self:check_ccmaster() then
	    local i = self.presetno and self.presetno or 1
	    i = val == 1 and i+1 or i-1
	    self:recall_preset(i)
	 end
	 return true
      elseif num == 0 and self.deck > 0 then
	 -- cross fade control
	 -- make sure that 64 gets mapped to the half-way value
	 local x = val==127 and 1.0 or val/128
	 if x ~= self.djdata.xfade then
	    self.djdata.xfade = x
	    -- adjust the volume
	    local val = self.djdata.vol[self.deck] * xfade(x)[self.deck]
	    -- round to integer
	    val = math.floor(val+0.5)
	    self.assert_master = true
	    return {val, 7, ch}
	 end
	 return true
      else
	 -- pass through
	 self.assert_master = not shift
	 return false
      end
   elseif self.deck == 0 or deck == self.deck then
      -- deck control, deck matches
      if num == 9 or num == 10 then
	 local i = param_i["pos"]
	 -- Raptor doesn't actually have a "CDJ" mode, if you want to speed
	 -- things up or slow them down, you'll have to use the tempo control
	 -- instead. Thus "vinyl" mode just indicates whether the jog wheel
	 -- is active (and scratching enabled) during playback (it's always
	 -- on when transport is stopped).
	 if i and (self.transport == 0 or self.djdata.vinyl[deck] ~= 0) then
	    -- scratch: true indicates scratch (top) mode, false normal (ring)
	    -- movement. These behave slightly differently on the DJ Control.
	    -- Specifically, ring movements run at two different speeds
	    -- depending on whether SHIFT is pressed (fast) or not (slow),
	    -- whereas top movements always seem to run at the same (fast)
	    -- speed. At least that's the case on my Inpulse 200 MK2, YMMV.
	    local scratch = num > 9
	    -- val=1 indicates forward, 127 backward motion
	    local delta = val == 1 and 1 or -1
	    -- Scale down the jog wheel a bit (10x by default). It's way too
	    -- fast for our purposes, since we're scrubbing beats, not
	    -- samples, so even tiny movements would otherwise make the
	    -- playback position jump around a lot. The default value of 10
	    -- seems to be about right for me, but you can adjust that value
	    -- using the djcontrol_scrub_factor variable above.
	    if delta == self.djdata.last_delta[deck] then
	       self.djdata.last_count[deck] = self.djdata.last_count[deck] + 1
	       if self.djdata.last_count[deck] >= djcontrol_scrub_factor then
		  self.djdata.last_count[deck] = 0
	       else
		  return true
	       end
	    else
	       self.djdata.last_count[deck] = 1
	       self.djdata.last_delta[deck] = delta
	       return true
	    end
	    -- we're moving, reset the stopped status
	    self.stopped = false
	    if self.arp.loopstate == 0 or self.transport == 0 then
	       -- set the playback position and/or anacrusis
	       local pos = self.pos + delta
	       -- clamp to the prescribed range
	       local min, max = params[i].min, params[i].max
	       -- pos has a nominal range of -24..24, but we also want to
	       -- clamp it to the actual number of beats
	       max = math.min(max, self.arp.beats)
	       min = -max
	       pos = math.min(max, math.max(min, pos))
	       if pos ~= self.pos then
		  self:set_pos(pos)
	       end
	    else
	       -- in loop mode, change the loop playback position instead
	       local pos = self.djdata.pos[deck] + delta
	       -- no need to clamp to any range here, since this value
	       -- never shows up on the GUI; we just take it modulo the
	       -- current loop size when setting the loop index
	       local n = math.min(#self.arp.loop, self.arp.loopsize)
	       self.djdata.pos[deck] = pos
	       self.arp:set_loopidx(pos % math.max(1, n))
	    end
	 end
	 return true
      elseif num == 0 and not shift then
	 -- volume slider (unshifted), coarse, mapped to CC7 (volume)
	 if self.deck > 0 then
	    self.djdata.vol[self.deck] = val
	    -- apply cross fade
	    val = val * xfade(self.djdata.xfade)[self.deck]
	    -- round to integer
	    val = math.floor(val+0.5)
	 end
	 self.assert_master = deck == self.deck
	 return {val, 7, ch}
      elseif num == 1 and not shift then
	 -- filter knob (unshifted), coarse, mapped to CC8 (balance)
	 self.assert_master = deck == self.deck
	 return {val, 8, ch}
      else
	 -- skip ccmaster check if already filtered by deck
	 self.assert_master = deck == self.deck
	 -- pass through
	 return false
      end
   else
      -- filtered out (other deck, or disabled)
      return true
   end
end

-- preprocessing of note and control data using the enabled control surfaces

function raptor:process_note(atoms)
   local ch = atoms[3] or 1
   local portno = (ch-1)//16+1
   -- The device drivers always listen on ports 2-4 only, so we can bypass the
   -- entire chain for all other port numbers.
   if portno >= 2 and portno <= 4 then
      -- Launchpad and APC mini only listen on ports 3+4.
      if portno == 3 or portno == 4 then
	 local res = apcmini ~= 0 and self:apcmini_note(atoms)
	 if res then
	    return res
	 end
	 res = launchpad ~= 0 and self:launchpad_note(atoms)
	 if res then
	    return res
	 end
      end
      -- Only port 2 gets processed from here on.
      if portno == 2 then
	 local res = launchkey ~= 0 and self:launchkey_note(atoms)
	 if res then
	    return res
	 end
	 res = launchcontrol ~= 0 and self:launchcontrol_note(atoms)
	 if res then
	    return res
	 end
	 res = midimix ~= 0 and self:midimix_note(atoms)
	 if res then
	    return res
	 end
	 res = pacer ~= 0 and self:pacer_note(atoms)
	 if res then
	    return res
	 end
	 -- always put djcontrol last since it also filters out messages, which
	 -- might interfere with the other controllers
	 res = djcontrol ~= 0 and self:djcontrol_note(atoms)
	 if res then
	    return res
	 end
      end
   end
   return false
end

function raptor:process_ctl(atoms)
   local ch = atoms[3] or 1
   local portno = (ch-1)//16+1
   -- The device drivers always listen on ports 2-4 only, so we can bypass the
   -- entire chain for all other port numbers.
   if portno >= 2 and portno <= 4 then
      -- Launchpad and APC mini only listen on ports 3+4.
      if portno == 3 or portno == 4 then
	 local res = apcmini ~= 0 and self:apcmini_ctl(atoms)
	 if res then
	    return res
	 end
	 res = launchpad ~= 0 and self:launchpad_ctl(atoms)
	 if res then
	    return res
	 end
      end
      -- Only port 2 gets processed from here on.
      if portno == 2 then
	 local res = launchkey ~= 0 and self:launchkey_ctl(atoms)
	 if res then
	    return res
	 end
	 local res = launchcontrol ~= 0 and self:launchcontrol_ctl(atoms)
	 if res then
	    return res
	 end
	 res = midimix ~= 0 and self:midimix_ctl(atoms)
	 if res then
	    return res
	 end
	 res = pacer ~= 0 and self:pacer_ctl(atoms)
	 if res then
	    return res
	 end
	 -- djcontrol is handled separately since it needs a different output
	 -- handling
      end
   end
   return false
end

function raptor:process_sysex(atoms, portno)
   -- only APCmini, Launchkey and Launchpad process sysex at this time
   local res = launchkey ~= 0 and self:launchkey_sysex(atoms, portno)
   if res then
      return res
   end
   local res = launchpad ~= 0 and self:launchpad_sysex(atoms, portno)
   if res then
      return res
   end
   local res = apcmini ~= 0 and self:apcmini_sysex(atoms, portno)
   if res then
      return res
   end
   return false
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
   local thru = ch <= 16 or ch > 16*self.thru
   if thru then
      return self.inchan == 0 or ch == self.inchan
   else
      -- skip MIDI data on secondary inputs (control surfaces and the like)
      return false
   end
end

function raptor:rechan(atoms)
   -- this should always be true, but if it isn't we simply use the channel
   -- information in the original message
   if self.chan > 0 then
      atoms[#atoms] = self.chan
   end
   return atoms
end

function raptor:in_1_thru(atoms)
   local thru = atoms[1]
   if type(thru) == "number" then
      -- this must be a nonnegative integer
      thru = math.max(0, math.floor(thru))
      self.thru = thru
   end
end

function raptor:in_1_note(atoms)
   local res = self:process_note(atoms)
   -- launchpad/launchkey: mapped note gets processed as if it was on input
   if res and type(res) ~= "table" then
      return
   end
   -- for the purposes of MIDI learn, notes are treated as if they were
   -- additional CCs starting at 128
   if self:check_midi_learn(atoms[2], atoms[1]+128, atoms[3]) or
      self:check_midi_map(atoms[2], atoms[1]+128, atoms[3]) then
      self.assert_master = false
      return
   end
   self.assert_master = false
   if self.bypass ~= 0 then
      -- pass through incoming notes (with transposition applied)
      if self:check_chan(atoms[3]) then
	 atoms[1] = atoms[1]+self.transp
	 self:outlet(1, "note", self:rechan(atoms))
      end
   else
      local num, val, ch = table.unpack(atoms)
      if not ch then
	 -- default channel
	 ch = 1
      end
      if type(num) == "number" and type(val) == "number" and type(ch) == "number" and self:check_chan(ch) then
	 if debug_level >= 4 then
	    if val > 0 then
	       print(string.format("[in] note on %d %d", num, val))
	    else
	       print(string.format("[in] note off %d", num))
	    end
	 end
	 self.arp:note(num, val)
	 self.chan = self:get_chan(ch)
	 if self.chan ~= 10 then
	    self.backup_chan = self.chan
	 end
      end
   end
end

-- other incoming MIDI messages (CC, pitch bend, etc.)

function raptor:in_1_ctl(atoms)
   local ch = atoms[3] or 1
   local portno = (ch-1)//16+1
   local res = self:process_ctl(atoms)
   if res and type(res) == "table" then
      -- launchpad/launchkey/apcmini: table result is mapped CC which gets
      -- processed as if it was on input
      if (portno == 2 or portno == 3) and res[2] == 7 and #res==3 then
	 -- Do some special-casing for the ninth fader on the Launchkey (49+)
	 -- and the APC mini mk2. We pass through CC7 here.
	 if self.assert_master or self:check_ccmaster() then
	    self:outlet(1, "ctl", self:rechan(res))
	 end
	 self.assert_master = false
	 return
      end
      goto skip
   elseif res then
      return
   end
   if portno == 2 then
      res = djcontrol ~= 0 and self:djcontrol_ctl(atoms)
      if type(res) == "table" and #res==3 then
	 -- djcontrol: mapped CC, passed through as if it was on input
	 if self.assert_master or self:check_ccmaster() then
	    self:outlet(1, "ctl", self:rechan(res))
	 end
	 self.assert_master = false
	 return
      elseif res then
	 return
      end
   end
   ::skip::
   if self:check_midi_learn(atoms[1], atoms[2], atoms[3]) or
      self:check_midi_map(atoms[1], atoms[2], atoms[3]) then
      self.assert_master = false
      return
   end
   self.assert_master = false
   -- simple pass-through; we do *not* check the MIDI channel here, but we do
   -- check for ccmaster to direct messages to the right instance
   if self:check_ccmaster() then
      if atoms[2] == 7 and self.deck > 0 then
	 -- volume CC, tie-in with cross fade control (djcontrol)
	 self.djdata.vol[self.deck] = atoms[1]
      end
      self:outlet(1, "ctl", self:rechan(atoms))
   end
end

function raptor:in_1_pgmset(atoms)
   if type(atoms[1]) == "number" then
      self.pgmset = atoms[1] ~= 0
   end
end

function raptor:in_1_pgm(atoms)
   -- kludge: this can be either an SMMF or a parameter set/get message, we
   -- deal with that here on the fly
   if #atoms > 1 then
      if self.pgmset then
	 -- pgmset mode: interpret program changes as preset switches, rather
	 -- than passing them on to a connected synth (this is disabled by
	 -- default and can be set with the init subpatch of the main patch)
	 if self:check_ccmaster() then
	    self:in_1_preset({atoms[1]})
	 end
      elseif self:check_ccmaster() then
	 -- similar to CC, we don't require the input channel to match, but we
	 -- do check ccmaster to direct the PC to the right instance(s)
	 self:outlet(1, "pgm", self:rechan(atoms))
      end
   else
      self:in_1("pgm", atoms)
   end
end

-- The following three methods deal with voice messages which are generally
-- associated with note data coming from a particular input, so at present we
-- filter them by input channel before passing them through. We also filter
-- them by ccmaster, though. Maybe doing both is overkill, but if a particular
-- Raptor instance was selected then that's probably what the user wants.

function raptor:in_1_bend(atoms)
   if self:check_chan(atoms[2]) and self:check_ccmaster() then
      -- vanilla-bug-compatible range adjustment needed here
      atoms[1] = atoms[1] - 8192
      self:outlet(1, "bend", self:rechan(atoms))
   end
end

function raptor:in_1_touch(atoms)
   if self:check_chan(atoms[2]) and self:check_ccmaster() then
      self:outlet(1, "touch", self:rechan(atoms))
   end
end

function raptor:in_1_polytouch(atoms)
   if self:check_chan(atoms[3]) and self:check_ccmaster() then
      atoms[2] = atoms[2]+self.transp
      self:outlet(1, "polytouch", self:rechan(atoms))
   end
end

function raptor:in_1_sysex(atoms)
   local res = self:process_sysex(atoms, self.portno)
   -- no pass-through here
end

function raptor:in_2_float(p)
   -- port number for sysex messages
   self.portno = math.floor(p)
end

-- instance parameters (these need special treatment)

function raptor:update_meter()
   -- update the meter in the arpeggiator
   self.arp:set_meter(self.n*self.division)
   -- we also need to update the loop size here
   local i = param_i["loopsize"]
   local loopsize = self.param_val[i]
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
      -- make sure to update the internal state as well, this will also
      -- update the panel
      self:param("meter-num", self.n)
      self:param("meter-denom", self.m)
      self:param("division", self.division)
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
      if debug_level >= 1 then
	 print(string.format("raptor (id %d) is up and running!", id))
	 -- this prints the names of all presets in the console
	 self:in_1_preset({})
      end
      -- we keep track of the id for various operations
      self.id = id
      -- we also keep track of the ids of all running raptor instances
      table.insert(raptor.instances, id)
      -- and the preset of each instance
      raptor.presets[id] = "default"
   end
end

-- midi learn (originally for CC only, but we can also use the same facility
-- for notes by mapping the note numbers to pseudo CCs 128-255)

function raptor:cctostring(cc, ch)
   if not cc then
      cc, ch = self.midi_learn_cc, self.midi_learn_ch
   end
   if cc < 128 then
      return cc_name(cc, ch)
   else
      return asa_pitch(cc-128, ch)
   end
end

function raptor:opttostring(opt)
   if opt == true then
      return " [toggle]"
   elseif type(opt) == "number" and opt < 0 then
      return " [inverted]"
   else
      return ""
   end
end

function raptor:load_map(fname2)
   local fname = fname2 and fname2 or self._canvaspath .. "data/" .. midimap_name
   local fp = io.open(fname, "r")
   if fp then
      local midi_map = fp:read("a")
      if midi_map then
	 local function check_map(midi_map)
	    if type(midi_map) ~= "table" then
	       return false
	    end
	    for cc, map in pairs(midi_map) do
	       if type(map) ~= "table" or type(cc) ~= "number" or math.floor(cc) ~= cc or cc < 0 or cc > 256 then
		  return false
	       end
	       for ch, v in pairs(map) do
		  local var, opt
		  if type(v) == "table" then
		     var, opt = table.unpack(v)
		  else
		     var, opt = v, nil
		  end
		  if type(var) ~= "string" or (opt and type(opt) ~= "number" and type(opt) ~= "boolean") or type(ch) ~= "number" or math.floor(ch) ~= ch or ch < 1 or ch > 128 then
		     return false
		  end
	       end
	    end
	    return true
	 end
	 local f = load("return " .. midi_map)
	 if type(f) == "function" then
	    midi_map = f()
	    -- do some quick plausability checks
	    if check_map(midi_map) then
	       if fname2 then
		  fp:close()
		  return midi_map
	       else
		  self.midi_map = midi_map
	       end
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
   local fname = self._canvaspath .. "data/" .. midimap_name
   local fp = io.open(fname, "w")
   if fp then
      fp:write(inspect(self.midi_map, { alttab = true }))
      fp:close()
      -- broadcast a message to all raptor instances so that they can update
      -- themselves
      pd.send("__raptor", "midimap", {})
   end
end

function raptor:map_get(cc, ch)
   local map = self.midi_map[cc]
   if map then
      local var = map[ch]
      if type(var) == "table" then
	 return table.unpack(var)
      else
	 return var
      end
   else
      return nil
   end
end

function raptor:map_set(cc, ch, var, opt)
   local map = self.midi_map[cc]
   if not map then
      map = {}
      self.midi_map[cc] = map
   end
   if opt then
      map[ch] = {var, opt}
   else
      map[ch] = var
   end
   -- launchpad/launchkey/apcmini tie-in: light up buttons on the grid when
   -- they're bound
   self:launchpad_mapped(cc, ch, var)
   self:launchkey_mapped(cc, ch, var)
   self:apcmini_mapped(cc, ch, var)
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

function raptor:learn(show)
   if self.midi_learn_cc and self.midi_learn_var then
      local i = param_i[self.midi_learn_var]
      local tgl = i and params[i].toggled and self.midi_learn_tgl or false
      local function sgn(x)
	 if x then return x>=0 and 1 or -1 else return nil end
      end
      local pol = i and not tgl and sgn(self.midi_learn_pol) or nil
      local var = self:map_get(self.midi_learn_cc, self.midi_learn_ch)
      self:map_set(self.midi_learn_cc, self.midi_learn_ch, self.midi_learn_var, tgl or pol)
      self:map_mode(0)
      print(string.format("%s %smapped to %s%s", self:cctostring(), var and "re" or "", self.midi_learn_var, self:opttostring(tgl or pol)))
      self:save_map()
      -- LP/LK feedback state
      self:update_state(true)
   elseif self.midi_learn_cc then
      local var = self:map_get(self.midi_learn_cc, self.midi_learn_ch)
      local tgl, pol = self.midi_learn_tgl, self.midi_learn_pol
      if var and show then
	 print(string.format("remapping %s%s currently mapped to %s, wiggle a control", self:cctostring(), self:opttostring(tgl or pol), var))
	 print("press learn again to abort, or press unlearn to unmap")
      elseif show then
	 print(string.format("mapping %s%s, wiggle a control", self:cctostring(), self:opttostring(tgl or pol)))
      end
   elseif self.midi_learn_var then
      local cc, ch = self:map_find(self.midi_learn_var)
      local tgl, pol = self.midi_learn_tgl, self.midi_learn_pol
      if cc and show then
	 print(string.format("mapping param %s%s already mapped to %s, send MIDI", self.midi_learn_var, self:opttostring(tgl or pol), self:cctostring(cc, ch)))
	 print("press learn again to abort, or press unlearn to unmap")
      elseif show then
	 print(string.format("mapping param %s%s, send MIDI", self.midi_learn_var, self:opttostring(tgl or pol)))
      end
   end
end

function raptor:check_midi_learn(val, cc, ch)
   if self.midi_learn == 1 then
      -- midi learn for CC
      local changed = self.midi_learn_cc ~= cc or self.midi_learn_ch ~= ch
      -- if pol_set is true then we don't touch the pol value any more, since
      -- it has been set already (or overridden with the tgl flag)
      local pol_set = self.midi_learn_var
      if not pol_set and changed then
	 -- need to reset the polarity state
	 self.midi_learn_val = nil
	 self.midi_learn_pol = nil
	 -- tgl needs to be reset as well
	 self.midi_learn_tgl = nil
      end
      if val > 0 and (changed or self.midi_learn_val ~= val) then
	 local function sgn(x) return x>=0 and 1 or -1 end
	 local pol = not pol_set and not self.midi_learn_tgl and self.midi_learn_val and val ~= self.midi_learn_val and sgn(val - self.midi_learn_val) or nil
	 self.midi_learn_cc = cc
	 self.midi_learn_ch = ch
	 self.midi_learn_val = val
	 if val == 127 then
	    -- switch to special toggle mode (in this case, rather than
	    -- controlling the value directly, the controller's off value is
	    -- ignored, and the on value toggles the existing value)
	    self.midi_learn_tgl = true
	 elseif pol and self.midi_learn_pol ~= pol then
	    changed = changed or self.midi_learn_pol or pol < 0
	    self.midi_learn_pol = pol
	 end
	 self:learn(changed)
	 return true
      end
   end
   return false
end

-- pickup check

function raptor:pickup_check(state)
   -- start (state == true) and stop (state == false) pickup checks
   self.check_pickup = state and pickup_mode ~= 0
end

function raptor:reset_pickup(var)
   -- reset the pickup state after param updates, unless a pickup check is
   -- currently in progress
   if not self.check_pickup and self.pickup_state and
      self.pickup_state.var == var then
      self.pickup_state = nil
   end
end

function raptor:pickup(cc, ch, var, state)
   -- check the pickup state of a CC change
   -- state denotes the current state (true iff the pickup check succeeded)
   -- if cc, ch, or var changes, state gives the new state of the check
   -- otherwise, the new state is true iff either the old or the new state is
   -- return the new state in either case
   if not self.pickup_state or self.pickup_state.var ~= var or
      self.pickup_state.cc ~= cc or self.pickup_state.ch ~= ch then
      self.pickup_state = { cc = cc, ch = ch, var = var, state = state }
   else
      state = self.pickup_state.state or state
      self.pickup_state.state = state
   end
   return state
end

-- apply an existing mapping, with pickup check

function raptor:from_midi(val, cc, ch)
   local var, opt = self:map_get(cc, ch)
   local tgl = opt==true
   local pol = not tgl and type(opt) == "number" and opt or 1
   if var then
      local i = param_i[var]
      if i then
	 if params[i].toggled then
	    -- eps value with min step width, 3rd return
	    local eps = 0
	    if tgl then
	       -- special toggle mode
	       if val > 0 then
		  return var, self.param_val[i] == 0 and 1 or 0, eps
	       else
		  return var
	       end
	    else
	       -- continuous controller, interpreted as toggle
	       return var, val > 0 and 1 or 0, eps
	    end
	 else
	    -- make sure that 64 gets mapped to the half-way value
	    local min, max = params[i].min, params[i].max
	    if var == "pos" then
	       -- this one is special, it has a nominal range of -24..24, but
	       -- we also want to clamp it to the actual number of beats
	       max = math.min(max, self.arp.beats)
	       min = -max
	    end
	    -- NOTE: We want these to "snap" to the min and max positions for
	    -- the 0 and 127 data bytes, respectively, to avoid strange
	    -- rounding issues with controller feedback.
	    if pol < 0 then
	       -- inverted
	       val = val==0 and max or val==127 and min or val/128*(min-max)+max
	    else
	       val = val==0 and min or val==127 and max or val/128*(max-min)+min
	    end
	    -- the eps values for the continous and integer range cases are
	    -- somewhat heuristic, might need some tuning
	    local eps = (max-min)/128
	    if int_param[i] then
	       val = math.floor(val)
	       eps = math.max(1, math.ceil(eps))
	    end
	    if self.check_pickup and cc < 128 then
	       -- check pickup value for CCs
	       if not self:pickup(cc, ch, var, eps == 0 or math.abs(self.param_val[i]-val) < eps) then
		  -- tie-in with Launchkey parameter display
		  self:launchkey_ccval(cc, ch, var, val, self.param_val[i])
		  return var
	       end
	    end
	    -- tie-in with Launchkey parameter display
	    self:launchkey_ccval(cc, ch, var, val)
	    return var, val, eps
	 end
	 return var
      end
   end
end

function raptor:check_midi_map(val, cc, ch)
   local var = self:map_get(cc, ch)
   if var and (self.assert_master or self:check_ccmaster(var)) then
      -- We don't do the pickup check for the Launchpad (ports 3+4), as its
      -- faders are by definition always in sync, and doing the pickup check
      -- would also interfere with the device feedback.
      self:pickup_check(ch <= 32 or ch > 64 or ch == 39) -- 39 == APC mini
      var, val = self:from_midi(val, cc, ch)
      if val then
	 -- apply existing mapping
	 self:param(var, val)
      end
      self:pickup_check(false)
      return true
   end
   return false
end

-- process MIDI learn messages

function raptor:in_1_learn()
   if self.midi_learn == 1 then
      print("MIDI learn mode aborted")
      self:map_mode(0)
   else
      self.midi_learn_cc = nil
      self.midi_learn_ch = nil
      self.midi_learn_var = nil
      self.midi_learn_val = nil
      self.midi_learn_pol = nil
      self.midi_learn_tgl = nil
      self:map_mode(1)
      print("MIDI learn mode, send MIDI or wiggle a control")
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
	    print(string.format("%s unmapped", self:cctostring()))
	    self:save_map()
	    done = true
	 end
      elseif self.midi_learn_var then
	 local cc, ch = self:map_find(self.midi_learn_var)
	 if cc then
	    self:map_set(cc, ch, nil)
	    print(string.format("%s unmapped", self:cctostring(cc, ch)))
	    self:save_map()
	    done = true
	 end
      end
      if done then
	 -- LP/LK feedback state
	 self:update_state(true)
      else
	 print("MIDI learn mode aborted")
      end
      self:map_mode(0)
   else
      self.midi_learn_cc = nil
      self.midi_learn_ch = nil
      self.midi_learn_var = nil
      self.midi_learn_val = nil
      self.midi_learn_pol = nil
      self.midi_learn_tgl = nil
      self:map_mode(1)
      print("MIDI learn mode, send MIDI or wiggle a control")
      print("press learn again to abort")
   end
end

function raptor:in_1_open_map()
   if self.id then
      local dir = self._canvaspath .. "data"
      pd.send(string.format("%s-open-map", self.id), "symbol", {dir})
   end
end

function raptor:in_1_merge_map(atoms)
   if type(atoms[1]) == "string" then
      local fname = atoms[1]
      local mmap = self:load_map(fname)
      if mmap then
	 local function merge(mmap)
	    local k, p, q = 0, 0, 0
	    for cc, map in pairs(mmap) do
	       for ch, v in pairs(map) do
		  k = k+1
		  local var, opt
		  if type(v) == "table" then
		     var, opt = table.unpack(v)
		  else
		     var, opt = v, nil
		  end
		  local var2, opt2 = self:map_get(cc, ch)
		  if var2 then
		     if var2 ~= var or opt2 ~= opt then
			p = p+1
			q = q+1
			print(string.format("%s remapped from %s%s to %s%s", self:cctostring(cc, ch), var2, self:opttostring(opt2), var, self:opttostring(opt)))
			self:map_set(cc, ch, var, opt)
		     end
		  else
		     p = p+1
		     self:map_set(cc, ch, var, opt)
		  end
	       end
	    end
	    return k, p, q
	 end
	 local k, p, q = merge(mmap)
	 print(string.format("added %d/%d mapping%s, %s conflict%s", p, k, p==1 and "" or "s", q>0 and tostring(q) or "no", q==1 and "" or "s"))
	 if p > 0 then
	    self:save_map()
	    -- LP/LK feedback state
	    self:update_state(true)
	 end
      else
	 self:error("couldn't load " .. fname)
      end
   end
end

-- switch between ccmasters

function raptor:in_1_ccmaster(atoms)
   local flag, id = table.unpack(atoms)
   if type(id) == "number" then
      id = string.format("%d", id)
   end
   if id and self.id then
      if flag == 0 then
	 -- launchpad/key fader page tie-in
	 self:launchpad_master_change(self.ccmaster, nil)
	 self:launchkey_master_change(self.ccmaster, nil)
	 self:apcmini_master_change(self.ccmaster, nil)
	 -- omni
	 self.ccmaster = nil
	 -- give feedback on the panel
	 pd.send(string.format("%s-ccmaster-status", self.id), "float", {0})
	 -- ccmaster feedback
	 self:djcontrol_ccmaster(0)
	 self:launchpad_ccmaster(0)
	 self:launchkey_ccmaster(0)
	 self:launchkey_ccmaster_state(0)
	 self:launchcontrol_ccmaster(0)
	 self:midimix_ccmaster(0)
	 self:apcmini_ccmaster(0)
      else
	 -- launchpad/key fader page tie-in
	 self:launchpad_master_change(self.ccmaster, id)
	 self:launchkey_master_change(self.ccmaster, id)
	 self:apcmini_master_change(self.ccmaster, id)
	 -- only the given raptor is receiving
	 self.ccmaster = id
	 -- give feedback on the panel
	 flag = self:check_ccmaster() and 1 or 0
	 pd.send(string.format("%s-ccmaster-status", self.id), "float", {flag})
	 -- ccmaster feedback
	 self:djcontrol_ccmaster(flag)
	 self:launchpad_ccmaster(flag)
	 self:launchkey_ccmaster(flag)
	 self:launchkey_ccmaster_state(flag)
	 self:launchcontrol_ccmaster(flag)
	 self:midimix_ccmaster(flag)
	 self:apcmini_ccmaster(flag)
      end
   else
      -- no ids, assume omni
      self.ccmaster = nil
   end
   if lk_select ~= 0 then
      self:launchkey_ccmaster_pads()
   end
end

function raptor:in_1_ccmaster_set(atoms)
   -- this message gets broadcast to all raptor instances, but only a single
   -- instance should respond to it
   if self.id and self.id == raptor.instances[1] then
      local i = atoms[1]
      if not i then
	 -- switch to omni if no argument
	 pd.send("all-arp", "ccmaster", {0, tonumber(self.id)})
      elseif type(i) == "number" then
	 local id = raptor.instances[i]
	 if not id or id == self.ccmaster then
	    -- switch to omni if no id, or if the given id already is the
	    -- ccmaster
	    pd.send("all-arp", "ccmaster", {0, tonumber(self.id)})
	 else
	    -- tell everyone about the new ccmaster
	    pd.send("all-arp", "ccmaster", {1, tonumber(id)})
	 end
      end
   end
end

function raptor:in_1_ccmaster_next()
   if self.id and self.id == raptor.instances[1] then
      local i = self.ccmaster and self:get_instance(self.ccmaster) or 0
      if i == 0 then
	 i = 1
      elseif i == #raptor.instances then
	 -- leaving i as is will return to omni mode so that we can wrap
	 -- around next time
      else
	 i = i % (#raptor.instances) + 1
      end
      self:in_1_ccmaster_set({i})
   end
end

function raptor:in_1_ccmaster_prev()
   if self.id and self.id == raptor.instances[1] then
      local i = self.ccmaster and self:get_instance(self.ccmaster) or 0
      if i == 0 then
	 i = #raptor.instances
      elseif i == 1 then
	 -- leaving i as is will return to omni mode so that we can wrap
	 -- around next time
      else
	 i = (i-2) % (#raptor.instances) + 1
      end
      self:in_1_ccmaster_set({i})
   end
end

-- switch between decks (djcontrol)

function raptor:in_1_deck(atoms)
   local deck = atoms[1]
   if type(deck) == "number" then
      -- must be integer, <0 means off, 0 means omni, >0 indicates deck number
      deck = math.floor(deck)
      -- we somewhat arbitrarily limit this to 16 decks here, most devices
      -- only have two, 1 = left, 2 = right
      deck = math.min(16, deck)
      self.deck = deck
      if self.id then
	 -- also keep track of assigned decks globally
	 raptor.decks[self.id] = deck>0 and deck or nil
      end
   end
end

-- locate an instance for a given deck by its index

function raptor:locate_deck_i(i, deck)
   if i and deck and deck > 0 then
      -- locate the ith instance with the given deck
      local function locate(k, deck)
	 for i, id in ipairs(raptor.instances) do
	    local d = raptor.decks[id]
	    if d and d == deck then
	       k = k-1
	       if k <= 0 then
		  return i
	       end
	    end
	 end
	 return nil
      end
      return locate(i, deck)
   else
      return i
   end
end

-- find the index per deck by instance (reversal of the above)

function raptor:locate_i_deck(i, deck)
   if i and deck and deck > 0 then
      -- count instances with the same deck
      local k = 0
      for j = 1, i do
	 local d = raptor.decks[raptor.instances[j]]
	 if d and d == deck then
	    k = k+1
	 end
      end
      -- assert k>0 (since raptor.decks[raptor.instances[i]] == deck, or we
      -- wouldn't be here)
      return k
   else
      return i
   end
end

-- transport

function raptor:in_1_transport(atoms)
   self.transport = atoms[1]
end

function raptor:in_1_master(atoms)
   local id = atoms[1]
   if type(id) == "number" then
      id = string.format("%d", id)
   elseif type(id) ~= "string" then
      return
   end
   -- launchpad/key fader page tie-in
   self:launchpad_master_change(time_master, id)
   self:launchkey_master_change(time_master, id)
   self:apcmini_master_change(time_master, id)
   time_master = id
end

-- djcontrol and launchpad tie-ins

function raptor:in_1_transport_state(atoms)
   self:djcontrol_play(atoms[1])
   self:launchpad_play(atoms[1])
   self:launchkey_play(atoms[1])
   self:apcmini_play(atoms[1])
end

function raptor:in_1_sync(atoms)
   local pos, loop_pos = atoms[1], atoms[2]
   -- reset the anacrusis
   self:set_pos(0)
   -- set the position in the bar
   self.arp:set_idx(pos)
   if loop_pos and self.transport ~= 0 and self.arp.loopstate ~= 0 then
      -- also set the loop position
      self.arp:set_loopidx(loop_pos)
   end
end

-- looper

function raptor:looper(name, cmd)
   if self.arp.loopstate == 1 then
      -- loop is playing, update meter and tempo information
      if self.division > 1 then
	 self.arp.loop.meter = {self.n, self.m, self.division}
      else
	 self.arp.loop.meter = {self.n, self.m}
      end
      self.arp.loop.tempo = self.tempo
   end
   local res, val = self.arp:loop_file(name, cmd)
   if res then
      if res == "loopsize" then
	 -- new loop was loaded, internal state is already updated, but we
	 -- still need to update panel and looper applet
	 if self.arp.loop.meter or self.arp.loop.tempo then
	    -- loop has meter and/or tempo data attached to it
	    if self.arp.loop.meter and type(self.arp.loop.meter) == "table" then
	       local n, m, division = table.unpack(self.arp.loop.meter)
	       if not division then
		  division = 1
	       end
	       if type(n) == "number" and type(m) == "number" and
		  type(division) == "number" then
		  n = math.max(1, math.floor(n))
		  m = math.max(1, math.floor(m))
		  division = math.max(1, math.floor(division))
	       end
	       local check = n*division ~= self.n*self.division
	       self.n = n
	       self.m = m
	       self.division = division
	       if check then
		  -- update the meter
		  self.arp:set_meter(self.n*self.division)
	       end
	       self:param("meter-num", self.n)
	       self:param("meter-denom", self.m)
	       self:param("division", self.division)
	    end
	    if self.arp.loop.tempo and type(self.arp.loop.tempo) == "number" then
	       local tempo = math.max(1, self.arp.loop.tempo)
	       if tempo ~= self.tempo then
		  self.tempo = tempo
		  self:param("tempo", self.tempo)
	       end
	    end
	 end
	 -- loopsize result
	 val = math.floor(val/self.arp.beats)
	 return res, val
      else
	 -- result of loop filename query
	 return res, val
      end
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
	    local last_loopstate = self.arp.loopstate
	    if var == "outchan" then
	       if v == 0 then
		  -- kludge: reset to the last non-drum channel we played on,
		  -- in order to not be stuck on channel 10
		  self.chan = self.backup_chan
		  v = self.chan
	       elseif v ~= 10 then
		  self.backup_chan = v
	       end
	    elseif var == "click" and v == 0 then
	       -- turn off the metronome click
	       self:metro_click(0, 0)
	    end
	    -- update the current value
	    self.param_val[i] = v
	    self:reset_pickup(var)
	    if self.param_set[i] == self.set then
	       -- these actually live in the raptor instance
	       self.param_set[i](self, var, v)
	    else
	       -- these all live in the arpeggiator
	       self.param_set[i](self.arp, v)
	    end
	    if self.id and not params[i].transport then
	       -- report changes to the panel
	       local id = self.id
	       pd.send(string.format("%s-%s", id, var), "set", {v})
	    end
	    if last_loopstate ~= self.arp.loopstate then
	       -- djcontrol and launchpad tie-in, updates the LOOP buttons
	       self:djcontrol_loop(self.arp.loopstate)
	       self:launchpad_loop(self.arp.loopstate)
	       self:launchkey_loop(self.arp.loopstate)
	       self:apcmini_loop(self.arp.loopstate)
	    end
	    -- launchpad fader bank feedback
	    self:launchpad_fader_val(var)
	    -- launchpad/launchkey pad feedback
	    self:launchpad_pad(var)
	    self:launchkey_pad(var)
	    self:apcmini_pad(var)
	 end
      end
   end
end

function raptor:in_1(sel, atoms)
   if sel == "loop" and type(atoms[1]) == "string" then
      -- loop file command, this needs special treatment
      local last_loopstate = self.arp.loopstate
      local name, cmd = table.unpack(atoms)
      -- default for cmd is 1 (save) if loop is playing, 0 (load) otherwise
      cmd = cmd or self.arp.loopstate
      -- synthetic parameters for MIDI learn
      sel = cmd==0 and "loop-load" or cmd==1 and "loop-save" or nil
      if self.midi_learn == 1 and sel and
	 self.midi_learn_var ~= sel and param_i[sel] then
	 self.midi_learn_var = sel
	 self.midi_learn_val = nil
	 self.midi_learn_pol = nil
	 self:learn(true)
      end
      local res, val = self:looper(name, cmd)
      if res then
	 self:outlet(1, res, {val})
      end
      if last_loopstate ~= self.arp.loopstate then
	 -- djcontrol and launchpad tie-in, updates the LOOP buttons
	 self:djcontrol_loop(self.arp.loopstate)
	 self:launchpad_loop(self.arp.loopstate)
	 self:launchkey_loop(self.arp.loopstate)
	 self:apcmini_loop(self.arp.loopstate)
      end
   else
      local i = param_i[sel]
      if self.midi_learn == 1 and atoms[1] and i then
	 local changed = self.midi_learn_var ~= sel
	 -- if pol_set is true then we don't touch the pol value any more,
	 -- since it has been set already; otherwise, if the parameter has
	 -- changed, we need to initialize the last value from the parameter
	 -- store
	 local pol_set = self.midi_learn_cc
	 if not pol_set and changed then
	    -- need to reset the polarity state
	    self.midi_learn_val = self.param_val[i]
	    self.midi_learn_pol = nil
	 end
	 if changed then
	    self.midi_learn_var = sel
	 end
	 if not pol_set and self.midi_learn_val and self.midi_learn_val ~= atoms[1] then
	    local function sgn(x) return x>=0 and 1 or -1 end
	    local pol = sgn(atoms[1] - self.midi_learn_val)
	    if self.midi_learn_pol ~= pol then
	       self.midi_learn_pol = pol
	       changed = true
	    end
	 end
	 self.midi_learn_val = atoms[1]
	 self:learn(changed)
      end
      self:param(sel, atoms[1])
   end
end
