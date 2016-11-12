require 'nn'

settings = {}

function settings.get_act(act)

  if act == 'ReLU' then return nn.ReLU()
  elseif act == 'ReLU6' then return nn.ReLU6()
  elseif act == 'PReLU' then return nn.PReLU()
  elseif act == 'RReLU' then return nn.RReLU()
  elseif act == 'ELU' then return nn.ELU()
  elseif act == 'LeakyReLU' then return nn.LeakyReLU()
  else return nn.ReLU() end

end


function settings.get_crit(crit)

  if crit == 'MSE' then return nn.MSECriterion():cuda()
  elseif crit == 'Abs' then return nn.AbsCriterion():cuda()
  elseif crit == 'Smooth' then return nn.SmoothL1Criterion():cuda()
  elseif crit == 'Dist' then return nn.DistKLDivCriterion():cuda()
  else return nn.MSECriterion():cuda() end

end

return settings
