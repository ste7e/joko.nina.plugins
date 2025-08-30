#region "copyright"

/*
    Copyright © 2021 - 2021 George Hilios <ghilios+NINA@googlemail.com>

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#endregion "copyright"

using KdTree;
using KdTree.Math;
using NINA.Core.Utility;
using NINA.Core.Utility.Notification;
using NINA.Image.Interfaces;
using NINA.Joko.Plugins.HocusFocus.AutoFocus;
using NINA.Joko.Plugins.HocusFocus.Interfaces;
using NINA.Joko.Plugins.HocusFocus.StarDetection;
using NINA.Joko.Plugins.HocusFocus.Utility;
using NINA.Profile.Interfaces;
using OpenCvSharp;
using OxyPlot.Series;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace NINA.Joko.Plugins.HocusFocus.Inspection {

    public class SensorDetectedStars {

        public SensorDetectedStars(
            double focuserPosition,
            HocusFocusStarDetectionResult starDetectionResult,
            IRenderedImage image) {
            this.FocuserPosition = focuserPosition;
            this.StarDetectionResult = starDetectionResult;
            this.Image = image;
        }

        public double FocuserPosition { get; private set; }
        public HocusFocusStarDetectionResult StarDetectionResult { get; private set; }
        public IRenderedImage Image { get; private set; }

        public override string ToString() {
            return $"{{{nameof(FocuserPosition)}={FocuserPosition.ToString()}, {nameof(StarDetectionResult)}={StarDetectionResult}}}";
        }
    }

    public class SensorParaboloidTiltHistoryModel {

        public SensorParaboloidTiltHistoryModel(
            int historyId,
            System.Drawing.Size imageSize,
            double pixelSizeMicrons,
            double fRatio,
            double focuserSizeMicrons,
            double finalFocusPosition,
            double tiltEffectMicrons,
            double curvatureEffectMicrons,
            double autoFocusOffset,
            TiltPlaneModel tiltPlaneModel,
            SensorParaboloidModel sensorModel) {
            HistoryId = historyId;
            ImageSize = imageSize;
            PixelSizeMicrons = pixelSizeMicrons;
            FRatio = fRatio;
            FocuserSizeMicrons = focuserSizeMicrons;
            FinalFocusPosition = finalFocusPosition;
            TiltEffectMicrons = tiltEffectMicrons;
            CurvatureEffectMicrons = curvatureEffectMicrons;
            AutoFocusOffset = autoFocusOffset;
            TiltPlaneModel = tiltPlaneModel;
            SensorModel = sensorModel;
        }

        public int HistoryId { get; private set; }
        public System.Drawing.Size ImageSize { get; private set; }
        public double PixelSizeMicrons { get; private set; }
        public double FRatio { get; private set; }
        public double FocuserSizeMicrons { get; private set; }
        public double FinalFocusPosition { get; private set; }
        public double TiltEffectMicrons { get; private set; }
        public double CurvatureEffectMicrons { get; private set; }
        public double AutoFocusOffset { get; private set; }
        public TiltPlaneModel TiltPlaneModel { get; private set; }
        public SensorParaboloidModel SensorModel { get; private set; }
    }

    public class SensorModel : BaseINPC {
        private readonly IProfileService profileService;
        private readonly IInspectorOptions inspectorOptions;
        private readonly IAutoFocusOptions autoFocusOptions;
        private readonly IAlglibAPI alglibAPI;
        private int nextHistoryId = 0;

        public SensorModel(IProfileService profileService, IInspectorOptions inspectorOptions, IAutoFocusOptions autoFocusOptions, IAlglibAPI alglibAPI) {
            this.profileService = profileService;
            this.inspectorOptions = inspectorOptions;
            this.autoFocusOptions = autoFocusOptions;
            this.alglibAPI = alglibAPI;
            SensorTiltHistoryModels = new AsyncObservableCollection<SensorParaboloidTiltHistoryModel>();
        }

        public Task UpdateModel(
            List<SensorDetectedStars> allDetectedStars,
            double fRatio,
            double focuserSizeMicrons,
            double finalFocusPosition,
            int stepSize,
            CancellationToken ct) {
            if (allDetectedStars.Count == 0) {
                throw new ArgumentException("Cannot update sensor model. No detected stars provided");
            }

            return Task.Run(() => {
                ModelLoaded = false;
                var firstStarDetectionResult = allDetectedStars.First().StarDetectionResult;
                var imageSize = firstStarDetectionResult.ImageSize;
                var pixelSize = firstStarDetectionResult.PixelSize;
                Logger.Info($"Building Sensor Model. FRatio ({fRatio}), Focuser Size ({focuserSizeMicrons}), Pixel Size ({pixelSize}), Image size ({imageSize})");

                var fitResult = RegisterStarsAndFit(allDetectedStars, pixelSize: pixelSize, focuserSizeMicrons: focuserSizeMicrons, imageSize: imageSize, stepSize: stepSize);
                var dataPoints = fitResult.Points;
                if (dataPoints.Count < 9) {
                    throw new Exception($"Need at least 9 registered stars. Found {dataPoints.Count}");
                }

                var sensorModelSolver = new SensorParaboloidSolver(
                    dataPoints: dataPoints,
                    sensorSizeMicronsX: imageSize.Width * pixelSize,
                    sensorSizeMicronsY: imageSize.Height * pixelSize,
                    inFocusMicrons: finalFocusPosition * focuserSizeMicrons,
                    fixedSensorCenter: inspectorOptions.FixedSensorCenter);
                var nlSolver = new NonLinearLeastSquaresSolver<SensorParaboloidSolver, SensorParaboloidDataPoint, SensorParaboloidModel>(this.alglibAPI);
                sensorModelSolver.PositiveCurvature = true;
                var positiveCurvatureSolution = nlSolver.SolveWinsorizedResiduals(sensorModelSolver, ct: ct);
                ct.ThrowIfCancellationRequested();
                positiveCurvatureSolution.EvaluateFit(nlSolver, sensorModelSolver);

                sensorModelSolver.PositiveCurvature = false;
                var negativeCurvatureSolution = nlSolver.SolveWinsorizedResiduals(sensorModelSolver, ct: ct);
                ct.ThrowIfCancellationRequested();
                negativeCurvatureSolution.EvaluateFit(nlSolver, sensorModelSolver);

                var solution = positiveCurvatureSolution.RMSErrorMicrons < negativeCurvatureSolution.RMSErrorMicrons ? positiveCurvatureSolution : negativeCurvatureSolution;
                Logger.Info($"Solved surface model: {solution}. RMS = {solution.RMSErrorMicrons:0.0000}, GoD: {solution.GoodnessOfFit:0.0000}, Stars: {solution.StarsInModel}");

                if (solution.GoodnessOfFit < 0.05) {
                    throw new Exception($"Sensor modeling failed. R² = {solution.GoodnessOfFit:#.00}");
                }

                DisplayedSensorModel = solution;
                SensorModelResult.Update(
                    solution,
                    imageSize,
                    pixelSizeMicrons: pixelSize,
                    fRatio: fRatio,
                    focuserStepSizeMicrons: focuserSizeMicrons,
                    finalFocusPosition: finalFocusPosition,
                    registeredStars: fitResult.RegisteredStars);

                var historyId = Interlocked.Increment(ref nextHistoryId);
                SensorTiltHistoryModels.Insert(0, new SensorParaboloidTiltHistoryModel(
                    historyId: historyId,
                    pixelSizeMicrons: pixelSize,
                    fRatio: fRatio,
                    imageSize: imageSize,
                    focuserSizeMicrons: focuserSizeMicrons,
                    finalFocusPosition: finalFocusPosition,
                    sensorModel: solution,
                    tiltEffectMicrons: SensorModelResult.TiltEffectMicrons,
                    curvatureEffectMicrons: SensorModelResult.CurvatureEffectMicrons,
                    autoFocusOffset: SensorModelResult.AutoFocusMeanOffset,
                    tiltPlaneModel: SensorModelResult.TiltPlaneModel));
                SelectedTiltHistoryModel = null;
                ModelLoaded = true;
            }, ct);
        }

        private List<SensorParaboloidDataPoint> ToInterpolatedGrid(
            List<SensorParaboloidDataPoint> dataPoints,
            System.Drawing.Size imageSize) {
            var allPointsTree = new KdTree<double, object>(2, new DoubleMath(), AddDuplicateBehavior.Error);
            foreach (var dataPoint in dataPoints) {
                allPointsTree.Add(new[] { dataPoint.X, dataPoint.Y }, null);
            }

            int starCount = 0;
            double totalDistance = 0.0d;
            foreach (var node in allPointsTree) {
                var nearestNeighbors = allPointsTree.GetNearestNeighbours(node.Point, 2);
                if (nearestNeighbors.Length < 2) {
                    continue;
                }

                var nearestPoint = nearestNeighbors[1].Point;
                var distance = Math.Sqrt((node.Point[0] - nearestPoint[0]) * (node.Point[0] - nearestPoint[0]) + (node.Point[1] - nearestPoint[1]) * (node.Point[1] - nearestPoint[1]));
                ++starCount;
                totalDistance += distance;
            }

            var meanDistance = totalDistance / starCount;
            var gridCellSize = meanDistance;
            var gridCellWidthCount = Math.Max(3, (int)(imageSize.Width / gridCellSize));
            if (gridCellWidthCount % 2 == 0) {
                // Ensure odd to cover the center point
                ++gridCellWidthCount;
            }
            var gridCellHeightCount = Math.Max(3, (int)(imageSize.Height / gridCellSize));
            if (gridCellHeightCount % 2 == 0) {
                // Ensure odd to cover the center point
                ++gridCellHeightCount;
            }

            alglib.rbfmodel model = null;
            alglib.rbfreport rep = null;
            try {
                this.alglibAPI.rbfcreate(2, 1, out model);

                double[,] xy = new double[dataPoints.Count, 3];
                for (int i = 0; i < dataPoints.Count; ++i) {
                    var dataPoint = dataPoints[i];
                    xy[i, 0] = dataPoint.X;
                    xy[i, 1] = dataPoint.Y;
                    xy[i, 2] = dataPoint.FocuserPosition;
                }

                alglib.rbfsetpoints(model, xy);
                double lambda, lambdaNS;
                if (this.inspectorOptions.InterpolationAmount == InterpolationAmountEnum.Small) {
                    lambda = 1.0e-6;
                    lambdaNS = 1.0e-6;
                } else if (this.inspectorOptions.InterpolationAmount == InterpolationAmountEnum.Medium) {
                    lambda = 1.0e-3;
                    lambdaNS = 1.0e-4;
                } else if (this.inspectorOptions.InterpolationAmount == InterpolationAmountEnum.Large) {
                    lambda = 1.0;
                    lambdaNS = 1.0e-2;
                } else {
                    throw new ArgumentException($"Interpolation Smoothing Amount {this.inspectorOptions.InterpolationAmount} not expected");
                }

                if (this.inspectorOptions.InterpolationAlgo == InterpolationAlgoEnum.Hierarchical) {
                    alglib.rbfsetalgohierarchical(model, meanDistance * 3.0d, 5, lambdaNS);
                } else if (this.inspectorOptions.InterpolationAlgo == InterpolationAlgoEnum.ThinPlateSpline) {
                    alglib.rbfsetalgothinplatespline(model, lambda);
                } else if (this.inspectorOptions.InterpolationAlgo == InterpolationAlgoEnum.MultiQuadric) {
                    alglib.rbfsetalgomultiquadricauto(model, lambda);
                } else if (this.inspectorOptions.InterpolationAlgo == InterpolationAlgoEnum.BiHarmonic) {
                    alglib.rbfsetalgobiharmonic(model, lambda);
                } else {
                    throw new ArgumentException($"InterpolationAlgo {this.inspectorOptions.InterpolationAlgo} not expected");
                }

                alglib.rbfbuildmodel(model, out rep);
                if (rep.terminationtype != 1) {
                    string reason;
                    if (rep.terminationtype == -5) {
                        reason = "non-distinct basis function centers were detected";
                    } else if (rep.terminationtype == -4) {
                        reason = "non-convergence";
                    } else if (rep.terminationtype == -3) {
                        reason = "incorrect model construction algorithm was chosen";
                    } else {
                        reason = "Unknown";
                    }
                    throw new Exception($"RBF interpolation failed with type {rep.terminationtype} and reason: {reason}");
                }

                double[] gridWidthNodes = new double[gridCellWidthCount];
                var gridWidthInterval = Math.Ceiling((double)imageSize.Width / (gridCellWidthCount - 1));
                var nextPosition = 0.0d;
                for (int i = 0; i < gridCellWidthCount; ++i) {
                    gridWidthNodes[i] = Math.Min(nextPosition, imageSize.Width - 1);
                    nextPosition += gridWidthInterval;
                }

                double[] gridHeightNodes = new double[gridCellHeightCount];
                var gridHeightInterval = Math.Ceiling((double)imageSize.Height / (gridCellHeightCount - 1));
                nextPosition = 0.0d;
                for (int i = 0; i < gridCellHeightCount; ++i) {
                    gridHeightNodes[i] = Math.Min(nextPosition, imageSize.Height - 1);
                    nextPosition += gridHeightInterval;
                }

                double[] outputNodes;
                alglib.rbfgridcalc2v(model, gridWidthNodes, gridWidthNodes.Length, gridHeightNodes, gridHeightNodes.Length, out outputNodes);

                var outputDataPoints = new List<SensorParaboloidDataPoint>();
                int outIdx = 0;
                for (int yIdx = 0; yIdx < gridCellHeightCount; ++yIdx) {
                    var y = gridHeightNodes[yIdx];
                    for (int xIdx = 0; xIdx < gridCellWidthCount; ++xIdx, ++outIdx) {
                        var x = gridWidthNodes[xIdx];
                        if (allPointsTree.RadialSearch(new double[] { x, y }, meanDistance, 1).Length > 0) {
                            outputDataPoints.Add(new SensorParaboloidDataPoint(x: x, y: y, focuserPosition: outputNodes[outIdx], rSquared: 1.0d));
                        }
                    }
                }
                return outputDataPoints;
            } finally {
                if (rep != null) {
                    this.alglibAPI.deallocateimmediately(ref rep);
                }
                if (model != null) {
                    this.alglibAPI.deallocateimmediately(ref model);
                }
            }
        }

        private RegistrationAndFitResult RegisterStarsAndFit(
            List<SensorDetectedStars> allDetectedStars,
            System.Drawing.Size imageSize,
            double focuserSizeMicrons,
            double pixelSize,
            int stepSize) {
            using (var stopwatch = MultiStopWatch.Measure()) {
                // registration phase

                int minHfrIndex = 0;
                double minHfr = allDetectedStars[0].StarDetectionResult.AverageHFR;
                for (int i = 1; i < allDetectedStars.Count; ++i) {
                    double nextHfr = allDetectedStars[i].StarDetectionResult.AverageHFR;
                    if (nextHfr < minHfr) {
                        minHfrIndex = i;
                        minHfr = nextHfr;
                    }
                }

                RegisteredStar[] registeredStars;

                if (autoFocusOptions.UseRANSAC) {
                    var targetStars = allDetectedStars[minHfrIndex].StarDetectionResult.StarList
                        .Select(s => new Point2D(s.Position.X, s.Position.Y, s.MaxBrightness))
                        .OrderBy(s => s.X + (s.Y * imageSize.Width));
                    double brightnessMinFactor = 0;
                    double maxBrightnessDiff = 0.5d;
                    double brightnessMinimumRef = allDetectedStars[minHfrIndex].StarDetectionResult.StarList.Average(s => s.MaxBrightness) * brightnessMinFactor;
                    for (int i = 0; i < allDetectedStars.Count; ++i) {
                        if (i == minHfrIndex) {
                            continue;
                        }
                        var theseStars = allDetectedStars[i].StarDetectionResult.StarList
                            .Select(s => new Point2D(s.Position.X, s.Position.Y, s.MaxBrightness))
                            .OrderBy(s => s.X + (s.Y * imageSize.Width));
                        double brightnessMinimum = allDetectedStars[i].StarDetectionResult.StarList.Average(s => s.MaxBrightness) * brightnessMinFactor;
                        var (putativeSrc, putativeDst) = RANSACRegistration.GeneratePutativeMatches(
                            theseStars.Where(s => s.Magnitude > brightnessMinimum).ToList(),
                            targetStars.Where(s => s.Magnitude > brightnessMinimumRef).ToList(),
                            maxDistance: 50, magnitudeDiffThreshold: maxBrightnessDiff);
                        //Trace.WriteLine($"Image {i}: From brightest {theseStars.Count(s => s.Magnitude > brightnessMinimum)} of {theseStars.Count()} stars found {putativeSrc.Count} putative matches against reference image {minHfrIndex}");
                        try {
                            // calculate the transform needed to register this image
                            var transform = RANSACRegistration.EstimateSimilarityTransform(putativeSrc, putativeDst);
                            //Trace.WriteLine($"Scale: {transform.Scale:F4}");
                            //Trace.WriteLine($"Rotation: {transform.Rotation:F4} radians");
                            //Trace.WriteLine($"Translation: ({transform.Tx:F4}, {transform.Ty:F4})");

                            // adjust each star according to the transform
                            for (int si = 0; si < allDetectedStars[i].StarDetectionResult.StarList.Count; si++) {
                                var oldPoint = allDetectedStars[i].StarDetectionResult.StarList[si].Position;
                                var transformedPoint = transform.Transform(new Point2D(allDetectedStars[i].StarDetectionResult.StarList[si].Position));
                                allDetectedStars[i].StarDetectionResult.StarList[si].Position = new Accord.Point((float)transformedPoint.X, (float)transformedPoint.Y);
                                //if (si < 5)
                                //    Trace.WriteLine($"Image {i}, star {si}, from {oldPoint} to {allDetectedStars[i].StarDetectionResult.StarList[si].Position}");
                            }
                        } catch (Exception ex) {
                            Trace.WriteLine($"Error: {ex.Message}");
                        }
                    }

                    // create dictionary of stars in the reference image
                    Dictionary<Point, List<MatchedStar>> starDict = new();
                    foreach (var star in allDetectedStars[minHfrIndex].StarDetectionResult.StarList) {
                        starDict.Add(new Point((int)star.Position.X, (int)star.Position.Y), new List<MatchedStar>() { new MatchedStar() {
                            FocuserPosition = allDetectedStars[minHfrIndex].FocuserPosition,
                            Star = (HocusFocusDetectedStar)star,
                            ImageIndex = minHfrIndex
                        } });
                    }

                    // in other images find stars that are close to the reference image
                    int searchSquareSide = 5;
                    for (int i = 0; i < allDetectedStars.Count; ++i) {
                        if (i == minHfrIndex) {
                            continue;
                        }
                        foreach (var star in allDetectedStars[i].StarDetectionResult.StarList) {
                            var searchPoint = new Point((int)star.Position.X, (int)star.Position.Y);
                            var searchArea = new System.Drawing.Rectangle(searchPoint.X - searchSquareSide, searchPoint.Y - searchSquareSide, searchSquareSide * 2 + 1, searchSquareSide * 2 + 1);
                            // find a star in the reference image that is within the search area and closest to the current star
                            IEnumerable<Point> refStars = starDict.Keys.Where(p =>
                                searchArea.Contains(p.X, p.Y)
                                    && (starDict[p].Average(d => d.Star.MaxBrightness) - star.MaxBrightness < maxBrightnessDiff)
                                    )
                                .OrderBy(p => DistanceSquared(p.X, p.Y, searchPoint.X, searchPoint.Y));
                            if (refStars.Count() > 0) {
                                starDict[refStars.Last()].Add(new MatchedStar() {
                                    FocuserPosition = allDetectedStars[i].FocuserPosition,
                                    Star = (HocusFocusDetectedStar)star,
                                    ImageIndex = i
                                });
                            }
                        }
                    }

                    float minBrightness = 0.05f;
                    float minMatchProportion = 0.75f;   // proportion of images that must have star for star to be included
                    int starCount = starDict.Keys.Count(pt => starDict[pt].Count >= allDetectedStars.Count * minMatchProportion);   // only use stars that are matched by 75% of the images
                    registeredStars = starDict.Keys.Where(pt => starDict[pt].Count >= allDetectedStars.Count * minMatchProportion).Select(pt => {
                        var rs = new RegisteredStar() {
                            RegistrationX = pt.X,
                            RegistrationY = pt.Y,
                        };
                        foreach (var focuserPositionGroup in starDict[pt].GroupBy(s => s.FocuserPosition)) {
                            var focuserPosition = focuserPositionGroup.Key;
                            MatchedStar aveStar = new MatchedStar() {
                                FocuserPosition = focuserPosition,
                                Star = new() {
                                    BoundingBox = focuserPositionGroup.First().Star.BoundingBox,
                                    Position = focuserPositionGroup.First().Star.Position,
                                    PSF = focuserPositionGroup.First().Star.PSF,
                                    Background = focuserPositionGroup.Average(g => g.Star.Background),
                                    AverageBrightness = focuserPositionGroup.Average(g => g.Star.Background),
                                    HFR = focuserPositionGroup.Average(g => g.Star.HFR),
                                    MaxBrightness = focuserPositionGroup.Average(g => g.Star.MaxBrightness),
                                }
                            };

                            rs.MatchedStars.Add(aveStar);
                        }
                        return rs;
                    }).Where(rs => rs.MatchedStars.Max(ms => ms.Star.MaxBrightness > minBrightness)).ToArray();

                    stopwatch.RecordEntry("RANSAC-based registration");

                    Trace.WriteLine($"Of all {starDict.Keys.Count} stars found, {starCount} were registered and matched to stars in other images.  These have been reduced to the brightest {registeredStars.Count()}");
                } else {
                    var allDetectedStarTrees = allDetectedStars.Select(result => {
                        var tree = new KdTree<float, DetectedStarIndex>(2, new FloatMath(), AddDuplicateBehavior.Error);
                        foreach (var (star, starIndex) in result.StarDetectionResult.StarList.Select((star, starIndex) => (star, starIndex))) {
                            tree.Add(new[] { star.Position.X, star.Position.Y }, new DetectedStarIndex(starIndex, (HocusFocusDetectedStar)star));
                        }
                        return tree;
                    }).ToArray();
                    stopwatch.RecordEntry("build trees");

                    float searchRadius = autoFocusOptions.UseRANSAC ? 3 : 30;
                    var globalRegistry = new KdTree<float, DetectedStarIndex>(2, new FloatMath(), AddDuplicateBehavior.Error);
                    var starIndexMap = Enumerable.Range(0, allDetectedStars.Count).Select(i => new Dictionary<int, int>()).ToArray();
                    foreach (var starNode in allDetectedStarTrees[minHfrIndex]) {
                        var nextIndex = globalRegistry.Count;
                        globalRegistry.Add(starNode.Point, new DetectedStarIndex(nextIndex, starNode.Value.DetectedStar));
                        starIndexMap[minHfrIndex].Add(starNode.Value.Index, nextIndex);
                    }

                    float[] pointDiff = new float[2];
                    for (int i = 0; i < allDetectedStars.Count; ++i) {
                        if (i == minHfrIndex) {
                            continue;
                        }

                        var nextStarList = allDetectedStars[i].StarDetectionResult.StarList;
                        var nextStarTree = allDetectedStarTrees[i];
                        var nextStarIndexMap = starIndexMap[i];
                        var matchedGlobalStars = new bool[globalRegistry.Count];
                        var matchedSourceStars = new bool[nextStarTree.Count];
                        var queue = new KdTree.PriorityQueue<MatchingPair, double>(new DoubleMath());
                        foreach (var (starNode, starNodeIndex) in nextStarTree.Select((starNode, starNodeIndex) => (starNode, starNodeIndex))) {
                            var sourceStar = starNode.Value.DetectedStar;
                            var sourcePoint = starNode.Point;
                            var sourceIndex = starNode.Value.Index;
                            var globalNeighbors = globalRegistry.RadialSearch(sourcePoint, searchRadius);
                            foreach (var globalNeighbor in globalNeighbors) {
                                var globalNeighborIndex = globalNeighbor.Value.Index;
                                pointDiff[0] = globalNeighbor.Point[0] - sourcePoint[0];
                                pointDiff[1] = globalNeighbor.Point[1] - sourcePoint[1];
                                var distance = MathUtility.DotProduct(pointDiff, pointDiff);
                                queue.Enqueue(new MatchingPair() { SourceIndex = sourceIndex, GlobalIndex = globalNeighborIndex }, distance);
                            }
                        }

                        while (queue.Count > 0) {
                            var nextCandidate = queue.Dequeue();
                            if (matchedGlobalStars[nextCandidate.GlobalIndex] || matchedSourceStars[nextCandidate.SourceIndex]) {
                                continue;
                            }

                            if (nextCandidate.SourceIndex != nextCandidate.GlobalIndex)
                                nextStarIndexMap.Add(nextCandidate.SourceIndex, nextCandidate.GlobalIndex);
                            matchedGlobalStars[nextCandidate.GlobalIndex] = true;
                            matchedSourceStars[nextCandidate.SourceIndex] = true;
                        }
                    }

                    registeredStars = new RegisteredStar[globalRegistry.Count];
                    foreach (var globalNode in globalRegistry) {
                        var registeredStar = new RegisteredStar() {
                            RegistrationX = globalNode.Value.DetectedStar.Position.X,
                            RegistrationY = globalNode.Value.DetectedStar.Position.Y
                        };
                        registeredStars[globalNode.Value.Index] = registeredStar;
                    }

                    for (int i = 0; i < starIndexMap.Length; ++i) {
                        var nextStarIndexMap = starIndexMap[i];
                        var focuserPosition = allDetectedStars[i].FocuserPosition;
                        var detectedStars = allDetectedStars[i].StarDetectionResult.StarList;

                        foreach (var nextKvp in nextStarIndexMap) {
                            var sourceIndex = nextKvp.Key;
                            var globalIndex = nextKvp.Value;
                            var sourceStar = (HocusFocusDetectedStar)detectedStars[sourceIndex];
                            var matchedStar = new MatchedStar() {
                                FocuserPosition = focuserPosition,
                                Star = sourceStar,
                                ImageIndex = i
                            };
                            registeredStars[globalIndex].MatchedStars.Add(matchedStar);
                        }
                    }
                }
                // registration phase done
                stopwatch.RecordEntry("registration");

                int discardedStarCount = 0;
                var sensorModelDataPoints = new List<SensorParaboloidDataPoint>();
                foreach (var registeredStar in registeredStars) {
                    if (registeredStar.MatchedStars.Count < 5) {
                        continue;
                    }

                    try {
                        var points = registeredStar.MatchedStars.Select(s => new ScatterErrorPoint(s.FocuserPosition, s.Star.HFR, 0.0d, 0.0d)).ToList();
                        AlglibHyperbolicFitting fitting;
                        if (autoFocusOptions.UnevenHyperbolicFitEnabled) {
                            fitting = HyperbolicUnevenFittingAlglib.Create(this.alglibAPI, points, stepSize, autoFocusOptions.WeightedHyperbolicFitEnabled);
                        } else {
                            fitting = HyperbolicFittingAlglib.Create(this.alglibAPI, points, autoFocusOptions.WeightedHyperbolicFitEnabled);
                        }

                        var solveResult = fitting.Solve();
                        if (!solveResult) {
                            Logger.Trace($"Failed to fit hyperbolic curve to star matches at ({registeredStar.RegistrationX:0.00}, {registeredStar.RegistrationY:0.00})");
                            discardedStarCount++;
                            continue;
                        }

                        if (fitting.RSquared < 0.90) {
                            // Discard bad fitting
                            discardedStarCount++;
                            continue;
                        }

                        var dataPointX = (registeredStar.RegistrationX - (imageSize.Width / 2.0)) * pixelSize;
                        var dataPointY = (registeredStar.RegistrationY - (imageSize.Height / 2.0)) * pixelSize;
                        var focuserMicrons = fitting.Minimum.X * focuserSizeMicrons;
                        var dataPoint = new SensorParaboloidDataPoint(dataPointX, dataPointY, focuserMicrons, fitting.RSquared);
                        sensorModelDataPoints.Add(dataPoint);
                    } catch (Exception e) {
                        Logger.Error(e, $"Failed to calculate hyperbolic at ({registeredStar.RegistrationX}, {registeredStar.RegistrationY}). Error={e.Message}");
                    }
                }

                stopwatch.RecordEntry("fitcurves");
                if (discardedStarCount > 0) {
                    Logger.Warning($"Discarded {discardedStarCount} stars during sensor modeling due to poor fits");
                }

                if (this.inspectorOptions.InterpolationEnabled && sensorModelDataPoints.Count > 5) {
                    return new RegistrationAndFitResult(ToInterpolatedGrid(sensorModelDataPoints, imageSize), registeredStars);
                } else {
                    return new RegistrationAndFitResult(sensorModelDataPoints, registeredStars);
                }
            }
        }

        private int DistanceSquared(int x1, int y1, int x2, int y2) {
            int xDist = x1 - x2;
            int yDist = y1 - y2;
            return (xDist * xDist + yDist * yDist);
        }

        private void UpdateTiltModels(SensorParaboloidTiltHistoryModel historyModel) {
            SensorModelResult.Update(
                sensorModel: historyModel.SensorModel, imageSize: historyModel.ImageSize, pixelSizeMicrons: historyModel.PixelSizeMicrons,
                fRatio: historyModel.FRatio, focuserStepSizeMicrons: historyModel.FocuserSizeMicrons, finalFocusPosition: historyModel.FinalFocusPosition,
                registeredStars: []);
            DisplayedSensorModel = historyModel.SensorModel;
        }

        private SensorParaboloidTiltHistoryModel selectedTiltHistoryModel;

        public SensorParaboloidTiltHistoryModel SelectedTiltHistoryModel {
            get => selectedTiltHistoryModel;
            set {
                try {
                    if (value != null) {
                        UpdateTiltModels(value);
                    }
                    selectedTiltHistoryModel = value;
                    RaisePropertyChanged();
                } catch (Exception e) {
                    Notification.ShowError($"Failed to set selected tilt history model. {e.Message}");
                    Logger.Error("Failed to set selected tilt history model", e);
                }
            }
        }

        public AsyncObservableCollection<SensorParaboloidTiltHistoryModel> SensorTiltHistoryModels { get; private set; }

        public SensorModelAberrationResult SensorModelResult { get; private set; } = new SensorModelAberrationResult();

        private SensorParaboloidModel displayedSensorModel;

        public SensorParaboloidModel DisplayedSensorModel {
            get => displayedSensorModel;
            private set {
                displayedSensorModel = value;
                RaisePropertyChanged();
            }
        }

        public void Reset() {
            this.ModelLoaded = false;
        }

        public void Clear() {
            SensorModelResult.Reset();
            SensorTiltHistoryModels.Clear();
            this.ModelLoaded = false;
        }

        #region Properties

        private bool modelLoaded = false;

        public bool ModelLoaded {
            get => modelLoaded;
            private set {
                if (modelLoaded != value) {
                    modelLoaded = value;
                    RaisePropertyChanged();
                }
            }
        }

        #endregion

        #region Private Classes

        private class DetectedStarIndex {

            public DetectedStarIndex(int index, HocusFocusDetectedStar star) {
                this.Index = index;
                this.DetectedStar = star;
            }

            public int Index { get; private set; }
            public HocusFocusDetectedStar DetectedStar { get; private set; }
        }

        private class MatchingPair {
            public int SourceIndex { get; set; }
            public int GlobalIndex { get; set; }
        }

        public class MatchedStar {
            public double FocuserPosition { get; set; }
            public HocusFocusDetectedStar Star { get; set; }
            public int ImageIndex { get; set; }

            public override string ToString() {
                return $"{{{nameof(FocuserPosition)}={FocuserPosition.ToString()}, {nameof(Star)}={Star}, {nameof(ImageIndex)}={ImageIndex}}}";
            }
        }

        public class RegisteredStar {
            public double RegistrationX { get; set; } = double.NaN;
            public double RegistrationY { get; set; } = double.NaN;
            public HyperbolicFittingAlglib Fitting { get; set; }
            public List<MatchedStar> MatchedStars { get; private set; } = new List<MatchedStar>();

            public override string ToString() {
                return $"{{{nameof(RegistrationX)}={RegistrationX.ToString()}, {nameof(RegistrationY)}={RegistrationY.ToString()}, {nameof(MatchedStars)}={MatchedStars}}}";
            }
        }

        #endregion
    }
}