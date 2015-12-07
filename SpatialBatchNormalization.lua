local BN, parent = nn.SpatialBatchNormalization, nn.Module

local empty = _.clone(parent.dpnn_mediumEmpty)
table.insert(empty, 'buffer')
table.insert(empty, 'buffer2')
table.insert(empty, 'centered')
table.insert(empty, 'std')
table.insert(empty, 'normalized')
table.insert(empty, 'output')
table.insert(empty, 'gradInput')
BN.dpnn_mediumEmpty = empty
