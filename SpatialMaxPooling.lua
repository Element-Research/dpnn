local SpatialMaxPooling, parent = nn.SpatialMaxPooling, nn.Module

local empty = _.clone(parent.dpnn_mediumEmpty)
table.insert(empty, 'indices')
SpatialMaxPooling.dpnn_mediumEmpty = empty
