require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

local metrics = require 'init'

aptest = {}

function aptest.TestAveragePrecision()
  
  local perm_test = function(y_true, y_score, ans)
    for i=1,5 do
      local perm = torch.randperm(y_true:size(1)):long()
      local ap = metrics.binary_average_precision(y_true:index(1, perm), y_score:index(1, perm))
      tester:assertle(math.abs(ap - ans), 1e-5)
    end
  end

  local y_true, y_score
  y_true = torch.Tensor{0, 0, 1, 1}
  y_score = torch.Tensor{0.1, 0.4, 0.35, 0.8}
  perm_test(y_true, y_score, 0.7916666666666667)

  y_true = torch.Tensor{0, 0, 0, 1, 1}
  y_score = torch.Tensor{0, 0, 0, 1, 1}
  perm_test(y_true, y_score, 1)

  y_true = torch.Tensor{1, 1, 1, 0, 0}
  y_score = torch.Tensor{0, 0, 0, 1, 1}
  perm_test(y_true, y_score, 0.3)

  -- random test case
  y_true = torch.Tensor{1, 1, 0, 0, 1, 0, 1, 1, 0, 1}
  y_score = torch.Tensor{0.3691809070336324, 0.8800287754245532, 0.011638727264165483, 0.006953687697083599, 0.11486090346118061, 0.32561174755094113, 0.11116869158442788, 0.8539578113194396, 0.631014307249777, 0.9426710493675786}
  perm_test(y_true, y_score, 0.86626984127)

  y_true = torch.Tensor{0, 0, 0, 0, 1, 1, 1, 0, 0, 0}
  y_score = torch.Tensor{0.5531595172441534, 0.8137851451389175, 0.9276554107778351, 0.1487818566151229, 0.07867739500148141, 0.2819462867791528, 0.621034912921882, 0.7253470152608638, 0.39249539494287267, 0.7409078543399679}
  perm_test(y_true, y_score, 0.185846560847)

  local timer = torch.Timer()
  y_true = torch.rand(20000):ge(0.5):float()
  y_score = torch.rand(20000)
  metrics.binary_average_precision(y_true, y_score)
  print('time used: ' .. timer:time().real)
end

tester = torch.Tester()
tester:add(aptest)
tester:run()
