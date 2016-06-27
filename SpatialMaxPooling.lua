local SpatialMaxPooling, parent = nn.SpatialMaxPooling, nn.Module
local _ = require 'moses'

local empty = _.clone(parent.dpnn_mediumEmpty)
table.insert(empty, 'indices')
SpatialMaxPooling.dpnn_mediumEmpty = empty
