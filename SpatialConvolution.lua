local SpatialConvolution, parent = nn.SpatialConvolution, nn.Module

local empty = _.clone(parent.dpnn_mediumEmpty)
table.insert(empty, 'finput')
table.insert(empty, 'fgradinput')
table.insert(empty, '_input')
table.insert(empty, '_gradOutput')
SpatialConvolution.dpnn_mediumEmpty = empty
