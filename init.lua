require 'torch'
local metrics = {}

-- Following scikit-learn implementation of average precision

-- Each row corresponds to one sample
-- Each column corresponds to one class
function metrics.average_precision(y_true, y_score)
  local num_class = y_true:size(2)
  local ans = 0
  for i=1,num_class do
    ans = ans + metrics.binary_average_precision(
                  y_true[{{}, {i}}]:squeeze(), y_score[{{}, {i}}]:squeeze())
  end
  return ans / num_class
end

function metrics.average_precision2(y_true, y_score)
  local num_class = y_true:size(2)
  local ans = 0
  for i=1,num_class do
    ans = ans + metrics.binary_average_precision2(
                  y_true[{{}, {i}}]:squeeze(), y_score[{{}, {i}}]:squeeze())
  end
  return ans / num_class
end


-- Compute binary average precision
--   y_true: 0/1 class labels
--   y_score: confidence value of predictions
function metrics.binary_average_precision(y_true, y_score)
  local precision, recall, thresholds = precision_recall_curve(y_true, y_score)
  return auc(recall, precision)
end

function auc(x, y)
  local sz = x:size(1)
  local direction
  if (x[{{2,sz}}]-x[{{1,sz-1}}]):ge(0):all() then
    direction = 1
  elseif (x[{{2,sz}}]-x[{{1,sz-1}}]):le(0):all() then
    direction = -1
  else
    error('x is not increasing')
  end
  -- np.trapz 
  local area2 = torch.cmul(y[{{2,sz}}]+y[{{1,sz-1}}], x[{{2,sz}}]-x[{{1,sz-1}}]):sum() / 2
  return direction * area2
end

function precision_recall_curve(y_true, proba_pred)
  local fps, tps, thresholds = binary_clf_curve(y_true, proba_pred)
  local precision = tps:clone():cdiv(tps + fps)
  local recall = tps / tps[-1]
  local last_ind = tps:eq(tps[-1]):nonzero()[1][1]
  local sl = torch.range(last_ind, 1, -1):long()
  return precision:index(1, sl):cat(torch.Tensor{1}),
         recall:index(1, sl):cat(torch.Tensor{0}),
         thresholds:index(1, sl)
end

function binary_clf_curve(y_true, y_score)
  assert(y_true:dim() == 1 and y_score:dim() == 1)
  assert(y_true:numel() == y_score:numel())

  local _, idx = torch.sort(y_score, true)
  y_score = y_score:index(1, idx)
  y_true = y_true:index(1, idx)

  local sz = y_true:size(1)
  local distinct_value_indices = (y_score[{{2,sz}}] - y_score[{{1,sz-1}}]):ne(0):nonzero()
  local threshold_idxs 
  if distinct_value_indices:nElement() == 0 then
    threshold_idxs = torch.LongTensor{sz}
  else
    threshold_idxs = distinct_value_indices:view(-1):cat(torch.LongTensor{sz})
  end

  local tps = y_true:cumsum():index(1, threshold_idxs)
  local fps = threshold_idxs:float() - tps
  return fps, tps, y_score:index(1, threshold_idxs)
end


------------- Alternate implementation -------------------
function metrics.binary_average_precision2(y_true, y_score)
  local _, idx = torch.sort(y_score, true)
  y_score = y_score:index(1, idx)
  y_true = y_true:index(1, idx)
  local score = 0
  local true_cnt = 0
  for i = 1,y_score:size(1) do
    if y_true[i] == 1 then
      true_cnt = true_cnt + 1
      score = score + true_cnt / i 
    end
  end
  return score / y_true:sum()
end

return metrics
