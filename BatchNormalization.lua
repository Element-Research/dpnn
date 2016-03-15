local BN, parent = nn.BatchNormalization, nn.Module

local empty = _.clone(parent.dpnn_mediumEmpty)
table.insert(empty, 'buffer')
table.insert(empty, 'buffer2')
table.insert(empty, 'centered')
table.insert(empty, 'std')
table.insert(empty, 'normalized')
table.insert(empty, 'output')
table.insert(empty, 'gradInput')
BN.dpnn_mediumEmpty = empty

-- for sharedClone
local params = _.clone(parent.dpnn_parameters)
table.insert(params, 'running_mean')
table.insert(params, 'running_var')
BN.dpnn_parameters = params
