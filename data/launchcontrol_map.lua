
-- This is my first attempt at an intuitive or at least a logical layout for
-- the LaunchControl XL. Enjoy. :) Albert Gräf <agggraef@gmail.com>, 2024

-- This maps the encoders in each column to the controls of the corresponding
-- row in the panel panel (1st column A = minvel, B = maxvel, Pan = velmod;
-- 2nd column = pmin, pmax, pmod, etc.). This works quite well, since the pan
-- pots are mapped to the mod controls in the third column of the panel, which
-- are mostly bipolar. The Send A and B controls are mostly mapped to ranges,
-- so that A controls the lower, B the upper value. Some rows have only a
-- single value (besides the mod value), in which case I mapped both A and B
-- to that value.

-- The faders are mapped as follows: 1 = hi, 2 = lo, 3 = transp, 4 = raptor
-- toggle, 5 = mode (random, up, down, etc.), 6 = loop size, 7 = loop toggle,
-- 8 = division. Note that the toggles will be engaged for any non-zero value,
-- so they can easily be turned on and off with just a tiny fader movement.
-- You might want to remap some or all of these, if you use other controls
-- more often. But note that the meter (numerator and denominator) aren't
-- mappable. (Presumably these might be controlled via plugin parameters when
-- running in a DAW, via plugdata. In the future the meter might also be
-- picked up through Ableton Link, but the current Raptor version doesn't
-- support this yet.)

{
  [13] = {
    [9] = "minvel"
  },
  [14] = {
    [9] = "pmin"
  },
  [15] = {
    [9] = "wmin"
  },
  [16] = {
    [9] = "gate"
  },
  [17] = {
    [9] = "hmin"
  },
  [18] = {
    [9] = "pref"
  },
  [19] = {
    [9] = "smin"
  },
  [20] = {
    [9] = "uniq"
  },
  [29] = {
    [9] = "maxvel"
  },
  [30] = {
    [9] = "pmax"
  },
  [31] = {
    [9] = "wmax"
  },
  [32] = {
    [9] = "gate"
  },
  [33] = {
    [9] = "hmax"
  },
  [34] = {
    [9] = "pref"
  },
  [35] = {
    [9] = "smax"
  },
  [36] = {
    [9] = "nmax"
  },
  [49] = {
    [9] = "velmod"
  },
  [50] = {
    [9] = "pmod"
  },
  [51] = {
    [9] = "gain"
  },
  [52] = {
    [9] = "gatemod"
  },
  [53] = {
    [9] = "hmod"
  },
  [54] = {
    [9] = "prefmod"
  },
  [55] = {
    [9] = "smod"
  },
  [56] = {
    [9] = "nmod"
  },
  [77] = {
    [9] = "pitchhi"
  },
  [78] = {
    [9] = "pitchlo"
  },
  [79] = {
    [9] = "transp"
  },
  [80] = {
    [9] = "raptor"
  },
  [81] = {
    [9] = "mode"
  },
  [82] = {
    [9] = "loopsize"
  },
  [83] = {
    [9] = "loop"
  },
  [84] = {
    [9] = "division"
  }
}
