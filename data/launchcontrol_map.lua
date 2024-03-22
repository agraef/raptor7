
-- This is my first attempt at an intuitive or at least a logical layout for
-- the LaunchControl XL. Enjoy. :) Albert Gräf <agggraef@gmail.com>, 2024

-- This maps the encoders in each column to the controls of the corresponding
-- row in the panel (1st column A = minvel, B = maxvel, Pan = velmod; 2nd
-- column = pmin, pmax, pmod, etc.). This works quite well, since the pan pots
-- are mapped to the mod controls in the third column of the panel, which are
-- mostly bipolar. The Send A and B controls are mostly mapped to ranges, so
-- that A controls the lower, B the upper value. Some rows have only a single
-- value (besides the mod value), in which case I mapped both A and B to that
-- value.

-- The faders are mapped as follows: 1 = hi, 2 = lo, 3 = mode (random, up,
-- down, etc.), 4 = pos/anacrusis (the "scrubbing" control), 5 = loop size,
-- 6 = meter numerator, 7 = meter denominator, 8 = division.

-- The send and track select buttons are mapped as follows:
-- up = rewind, down = raptor toggle, left = loop, right = play.

-- Finally, the mute, solo, and record arm buttons are mapped to the mute,
-- latch, and bypass toggles in the panel.

-- I think that this covers the most important functions that you might want
-- to use during a live performance. Of course, YMMV, so feel free to change
-- these around to better suit your style.

{
  [13] = { [25] = "minvel" },
  [14] = { [25] = "pmin" },
  [15] = { [25] = "wmin" },
  [16] = { [25] = "gate" },
  [17] = { [25] = "hmin" },
  [18] = { [25] = "pref" },
  [19] = { [25] = "smin" },
  [20] = { [25] = "uniq" },
  [29] = { [25] = "maxvel" },
  [30] = { [25] = "pmax" },
  [31] = { [25] = "wmax" },
  [32] = { [25] = "gate" },
  [33] = { [25] = "hmax" },
  [34] = { [25] = "pref" },
  [35] = { [25] = "smax" },
  [36] = { [25] = "nmax" },
  [49] = { [25] = "velmod" },
  [50] = { [25] = "pmod" },
  [51] = { [25] = "gain" },
  [52] = { [25] = "gatemod" },
  [53] = { [25] = "hmod" },
  [54] = { [25] = "prefmod" },
  [55] = { [25] = "smod" },
  [56] = { [25] = "nmod" },
  [77] = { [25] = "pitchhi" },
  [78] = { [25] = "pitchlo" },
  [79] = { [25] = "mode" },
  [80] = { [25] = "pos" },
  [81] = { [25] = "loopsize" },
  [82] = { [25] = "meter-num" },
  [83] = { [25] = "meter-denom" },
  [84] = { [25] = "division" },
  [104] = { [25] = { "rewind", true } },
  [105] = { [25] = { "raptor", true } },
  [106] = { [25] = { "loop", true } },
  [107] = { [25] = { "play", true } },
  [234] = { [25] = { "mute",true } },
  [235] = { [25] = { "latch",true } },
  [236] = { [25] = { "bypass",true } }
}
