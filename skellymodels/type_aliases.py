TrackedPointName = str
KeypointName = str
SegmentName = str
DimensionName = str

TrackedPointList = list[TrackedPointName]
WeightedTrackedPoints = dict[TrackedPointName, float]
KeypointMappingType = TrackedPointName | TrackedPointList | WeightedTrackedPoints
