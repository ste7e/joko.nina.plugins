#region "copyright"

/*
    Copyright © 2021 - 2021 George Hilios <ghilios+NINA@googlemail.com>

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#endregion "copyright"

using Accord.Statistics.Models.Fields;
using KdTree;
using KdTree.Math;
using NINA.Core.Model;
using NINA.Core.Utility;
using NINA.Core.Utility.Notification;
using NINA.Image.ImageAnalysis;
using NINA.Image.Interfaces;
using NINA.Joko.Plugins.HocusFocus.AutoFocus;
using NINA.Joko.Plugins.HocusFocus.Interfaces;
using NINA.Joko.Plugins.HocusFocus.StarDetection;
using NINA.Joko.Plugins.HocusFocus.Utility;
using NINA.Profile.Interfaces;
using NINA.WPF.Base.Utility.AutoFocus;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using OxyPlot.Series;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Speech.Recognition;
using System.Threading;
using System.Threading.Tasks;
using static System.Windows.Forms.AxHost;

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
            IProgress<ApplicationStatus> progress,
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

                var (solution, fitResult) = RegisterStarsAndFit(allDetectedStars,
                    pixelSize: pixelSize,
                    focuserSizeMicrons: focuserSizeMicrons,
                    finalFocusPosition: finalFocusPosition,
                    imageSize: imageSize,
                    stepSize: stepSize,
                    progress: progress,
                    ct: ct);

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

        private SensorParaboloidModel FitParaboloidModel(
            double focuserSizeMicrons,
            double finalFocusPosition,
            System.Drawing.Size imageSize,
            double pixelSize,
            RegistrationAndFitResult fitResult,
            CancellationToken ct,
            IProgress<ApplicationStatus> progress) {
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
            var positiveCurvatureSolution = nlSolver.SolveWinsorizedResiduals(sensorModelSolver, ct: ct, progress: progress);
            ct.ThrowIfCancellationRequested();
            positiveCurvatureSolution.EvaluateFit(nlSolver, sensorModelSolver);

            sensorModelSolver.PositiveCurvature = false;
            var negativeCurvatureSolution = nlSolver.SolveWinsorizedResiduals(sensorModelSolver, ct: ct, progress: progress);
            ct.ThrowIfCancellationRequested();
            negativeCurvatureSolution.EvaluateFit(nlSolver, sensorModelSolver);

            var solution = positiveCurvatureSolution.RMSErrorMicrons < negativeCurvatureSolution.RMSErrorMicrons ? positiveCurvatureSolution : negativeCurvatureSolution;
            Logger.Info($"Solved surface model: {solution}. RMS = {solution.RMSErrorMicrons:0.0000}, GoD: {solution.GoodnessOfFit:0.0000}, Stars: {solution.StarsInModel}");

            return solution;
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

        private enum IterationDirection { Up, Down };

        private (SensorParaboloidModel, RegistrationAndFitResult) RegisterStarsAndFit(
            List<SensorDetectedStars> allDetectedStars,
            System.Drawing.Size imageSize,
            double focuserSizeMicrons,
            double finalFocusPosition,
            double pixelSize,
            IProgress<ApplicationStatus> progress,
            int stepSize,
            CancellationToken ct) {
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
                ct.ThrowIfCancellationRequested();

                RegisteredStar[] registeredStars = null;

                // set relative brightness level for each star in each image
                foreach (var detectedStars in allDetectedStars) {
                    double imageMaxBrightness = detectedStars.StarDetectionResult.StarList.Max(s => s.AverageBrightness);
                    double imageMinBrightness = detectedStars.StarDetectionResult.StarList.Min(s => s.AverageBrightness);
                    foreach (var (star, index) in detectedStars.StarDetectionResult.StarList
                                                                    .Select((star, index) => ((HocusFocusDetectedStar)star, index))) {
                        star.NormalisedBrightness = (float)((star.AverageBrightness - imageMinBrightness) / (imageMaxBrightness - imageMinBrightness));
                    }
                }
                stopwatch.RecordEntry("normalise brightness");
                ct.ThrowIfCancellationRequested();

                int ransacAligned = 0;
                if (inspectorOptions.UseRANSAC) {
                    ransacAligned = AlignStarsWithRANSAC(allDetectedStars, imageSize, stopwatch, minHfrIndex, progress);
                    if (ransacAligned == allDetectedStars.Count) {
                        Trace.WriteLine("Ransac failed, reverting to non-aligned processing");
                    }
                }
                ct.ThrowIfCancellationRequested();

                double maxNormalisedBrightnessDiff = (inspectorOptions.StartingBrightnessDiff != -1) ? inspectorOptions.StartingBrightnessDiff : inspectorOptions.PreviousRunBrightnessDiff;
                SensorParaboloidModel bestPfit = null;
                RegistrationAndFitResult bestReg = null;
                double bestBrightnessDiff = maxNormalisedBrightnessDiff;

                Dictionary<double, (SensorParaboloidModel, RegistrationAndFitResult)> previousRuns = new();
                IterationDirection direction = IterationDirection.Up;
                bool upExhausted = false;
                bool downExhausted = false;
                SensorParaboloidModel prevFit = null;
                RegistrationAndFitResult prevReg = null;

                Trace.WriteLine($"Starting fit loop with previous MNBD of {maxNormalisedBrightnessDiff:#.##}");

                int iterations = 0;
                for (bool retry = true; retry;) {
                    var startTime = DateTime.Now;
                    retry = false;
                    int rejectionsOnBrightnessDiff;
                    if (inspectorOptions.UseTrees) {
                        (registeredStars, rejectionsOnBrightnessDiff) = MatchStarsUsingKdTree(allDetectedStars,
                            stopwatch, minHfrIndex,
                            (ransacAligned == allDetectedStars.Count) ? 3 : 30,
                            inspectorOptions.RejectBadBrightnessMatches ? maxNormalisedBrightnessDiff : -1,
                            progress);
                    } else {
                        (registeredStars, rejectionsOnBrightnessDiff) = MatchStarsWithoutKdTree(allDetectedStars, imageSize,
                            stopwatch, minHfrIndex,
                            (ransacAligned == allDetectedStars.Count) ? 3 : 30,
                            inspectorOptions.RejectBadBrightnessMatches ? maxNormalisedBrightnessDiff : -1,
                            progress);
                    }

                    // registration phase done
                    stopwatch.RecordEntry("registration");
                    ct.ThrowIfCancellationRequested();

                    RegistrationAndFitResult reg = FitImages(imageSize, focuserSizeMicrons, pixelSize, stepSize, stopwatch, registeredStars, progress, inspectorOptions.RejectBadlyFittingMatches);
                    SensorParaboloidModel pfit = null;

                    if (reg.Points.Count >= 9) {  // 9 points is the minimum for fitting the model
                        pfit = FitParaboloidModel(focuserSizeMicrons, finalFocusPosition, imageSize, pixelSize, reg, ct, progress);
                    } else {
                        retry = true;   // try again at a different brightness tolerance
                    }

                    if (inspectorOptions.RejectBadBrightnessMatches) {
                        double targetR2 = TargetR2BasedOnTimeTaken(DateTime.Now - startTime);  // an R2 of greater than this value will be acceptible and halt further iterations
                        Trace.WriteLine($"Target R2 {targetR2:#.##}");
                        if ((pfit != null) && ((pfit.StarsInModel < 10) || (pfit.GoodnessOfFit < targetR2) || IsFitTooGood(pfit))) {
                            Trace.WriteLine($"Only have {pfit.StarsInModel} stars and R2 of {pfit.GoodnessOfFit:#.##} (target is {targetR2:#.##}) with a brightness tolerance of {maxNormalisedBrightnessDiff:#.##}");
                            retry = true;
                        }

                        if (IsBetterFit(pfit, bestPfit)) {
                            bestPfit = pfit;
                            bestReg = reg;
                            bestBrightnessDiff = maxNormalisedBrightnessDiff;
                        }

                        if (inspectorOptions.RejectBadBrightnessMatches) {
                            previousRuns.Add(maxNormalisedBrightnessDiff, (pfit, reg));

                            if (retry) {    // keep going in the same direction
                                Trace.WriteLine($"this brightness tolerance={maxNormalisedBrightnessDiff:#.##}, direction: {direction}");
                                if (direction == IterationDirection.Up) {
                                    if ((maxNormalisedBrightnessDiff >= 3) || (rejectionsOnBrightnessDiff == 0)) {    // if there're no rejections at the current level don't increase tolerance
                                        upExhausted = true;
                                    }
                                    if ((pfit != null && !IsFitTooGood(pfit) && (!IsBetterFit(pfit, prevFit)) || upExhausted)) { // change direction unless already exhausted
                                        if (downExhausted) { // all tried, no retry
                                            retry = false;
                                        } else {
                                            if (maxNormalisedBrightnessDiff != inspectorOptions.PreviousRunBrightnessDiff)
                                                upExhausted = true;
                                            maxNormalisedBrightnessDiff = inspectorOptions.PreviousRunBrightnessDiff / 1.5;
                                            direction = IterationDirection.Down;
                                            Trace.WriteLine($"Switching direction to {direction}");
                                        }
                                    } else {
                                        maxNormalisedBrightnessDiff *= 1.5;
                                        direction = IterationDirection.Up;
                                    }
                                } else {
                                    if (maxNormalisedBrightnessDiff <= 0.01d)
                                        downExhausted = true;
                                    if ((pfit != null && !IsFitTooGood(pfit) && (!IsBetterFit(pfit, prevFit)) || downExhausted)) { // need to switch direction unless already exhausted
                                        if (upExhausted) { // all tried, no retry
                                            retry = false;
                                        } else {
                                            if (maxNormalisedBrightnessDiff != inspectorOptions.PreviousRunBrightnessDiff)
                                                downExhausted = true;
                                            maxNormalisedBrightnessDiff = inspectorOptions.PreviousRunBrightnessDiff * 1.5;
                                            direction = IterationDirection.Up;
                                            Trace.WriteLine($"Switching direction to {direction}");
                                        }
                                    } else {
                                        maxNormalisedBrightnessDiff /= 1.5;
                                        direction = IterationDirection.Down;
                                    }
                                }
                                if (retry) {
                                    Trace.WriteLine($"next brightness tolerance={maxNormalisedBrightnessDiff:#.##}, direction: {direction}");
                                } else {
                                    Trace.WriteLine("No more iterations");
                                }
                            }
                        }

                        prevFit = pfit;
                        prevReg = reg;
                        iterations++;

                        Trace.WriteLine($"End of registerStarsAndFit iteration with R2 of {bestPfit?.GoodnessOfFit:#.##}, retry is {retry}");
                    } else {        // we're not trying to find a better return as we're not rejecting on brightness
                        bestPfit = pfit;
                        bestReg = reg;
                        bestBrightnessDiff = maxNormalisedBrightnessDiff;
                    }
                }

                progress.Report(new ApplicationStatus());

                Trace.WriteLine($"After {iterations} iterations best fit is {bestPfit?.GoodnessOfFit:#.##}");

                inspectorOptions.PreviousRunBrightnessDiff = bestBrightnessDiff;
                previousRuns.Select(r => (r.Key, r.Value.Item1?.GoodnessOfFit, r.Value.Item1?.StarsInModel))
                    .ToList()
                    .ForEach(run => Trace.WriteLine($"Previous run: brightness tolerance {run.Key:#.##}, R2 {run.GoodnessOfFit:#.##}, Stars {run.StarsInModel} best? {run.Key == bestBrightnessDiff}"));

                if (bestPfit == null) {
                    throw new Exception("Failed to find a good model.");
                }
                if (bestPfit.GoodnessOfFit < 0.05) {
                    throw new Exception($"Sensor modeling failed. R² = {bestPfit.GoodnessOfFit:#.00}");
                }

                return (bestPfit, bestReg);
            }
        }

        private double TargetR2BasedOnTimeTaken(TimeSpan timeSpan) {
            double secondsTaken = timeSpan.TotalSeconds;
            if (secondsTaken < 1) {     // iterations are quick so we'll be very demanding
                return 0.9;
            }
            if (secondsTaken < 3) {
                return 0.8;
            }
            if (secondsTaken < 5) {
                return 0.7;
            }
            return 0.6; // iterations are slow so we'll set a low target
        }

        private static bool IsFitTooGood(SensorParaboloidModel fit) {
            if (1 - fit.GoodnessOfFit < 0.005) { // 1 means too good a fit - probably not enough points
                return true;
            } else {
                return false;
            }
        }

        private static bool IsBetterFit(SensorParaboloidModel thisFit, SensorParaboloidModel otherFit) {
            if (thisFit == null) {
                return false;
            }
            if (otherFit == null) {
                return true;
            }

            if (otherFit.GoodnessOfFit == 0) {
                return true;
            }
            if (thisFit.GoodnessOfFit == 0) {
                return false;
            }

            if (IsFitTooGood(thisFit)) { // 1 means too good a fit - probably not enough points
                return false;
            }
            if (IsFitTooGood(otherFit)) {
                return true;
            }
            return (thisFit.GoodnessOfFit >= otherFit.GoodnessOfFit);
        }

        private static List<StarDetectionRegion> CreateFullRegionSet(System.Drawing.Size imageSize, int rows, int cols) {
            var xPart = 1.0d / (double)cols;
            var yPart = 1.0d / (double)rows;
            var width = xPart * imageSize.Width;
            var height = yPart * imageSize.Height;
            var regions = new List<StarDetectionRegion>();
            int index = 0;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    regions.Add(new StarDetectionRegion(new RatioRect(c * xPart, r * yPart, width, height), ++index));
            return regions;
        }

        private RegistrationAndFitResult FitImages(
                System.Drawing.Size imageSize,
                double focuserSizeMicrons,
                double pixelSize,
                int stepSize,
                MultiStopWatch stopwatch,
                RegisteredStar[] registeredStars,
                IProgress<ApplicationStatus> progress,
                bool rejectBadlyFittingMatches) {
            int discardedStarCount = 0;
            var sensorModelDataPoints = new List<SensorParaboloidDataPoint>();
            var maxOutlierRejectedPoints = this.autoFocusOptions.MaxOutlierRejections;
            var rejectionConfidence = this.autoFocusOptions.OutlierRejectionConfidence;
            const int minStarCountForFitting = 5;
            int totalRejectedPointCount = 0;
            foreach (var registeredStar in registeredStars) {
                if (registeredStar.MatchedStars.Count < minStarCountForFitting) {
                    continue;
                }

                try {
                    var points = registeredStar.MatchedStars.Select(s => new ScatterErrorPoint(s.FocuserPosition, s.Star.HFR, 0.0d, 0.0d)).ToList();
                    var rejectedPoints = new List<ScatterErrorPoint>();
                    bool continueFitting;
                    AlglibHyperbolicFitting fitting;
                    bool solveResult;
                    do {
                        continueFitting = false;
                        if (autoFocusOptions.UnevenHyperbolicFitEnabled) {
                            fitting = HyperbolicUnevenFittingAlglib.Create(this.alglibAPI, points, stepSize, autoFocusOptions.WeightedHyperbolicFitEnabled);
                        } else {
                            fitting = HyperbolicFittingAlglib.Create(this.alglibAPI, points, autoFocusOptions.WeightedHyperbolicFitEnabled);
                        }

                        solveResult = fitting.Solve();
                        if (rejectBadlyFittingMatches) {
                            if (solveResult && rejectedPoints.Count < maxOutlierRejectedPoints && points.Count > minStarCountForFitting) {
                                var rejectedPoint = MathUtility.RejectionTest(points: points, fitting: fitting.Fitting, confidence: rejectionConfidence);
                                if (rejectedPoint != null) {
                                    rejectedPoints.Add(rejectedPoint);
                                    points.Remove(rejectedPoint);
                                    continueFitting = true;
                                }
                            }
                        }
                    } while (continueFitting);

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

                    totalRejectedPointCount += rejectedPoints.Count;
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
            if (totalRejectedPointCount > 0) {
                Logger.Info($"Rejected {totalRejectedPointCount} points while fitting {sensorModelDataPoints.Count} stars");
            }

            if (this.inspectorOptions.InterpolationEnabled && sensorModelDataPoints.Count > 5) {
                return new RegistrationAndFitResult(ToInterpolatedGrid(sensorModelDataPoints, imageSize), registeredStars);
            } else {
                return new RegistrationAndFitResult(sensorModelDataPoints, registeredStars);
            }
        }

        private int AlignStarsWithRANSAC(
            List<SensorDetectedStars> allDetectedStars,
            System.Drawing.Size imageSize,
            MultiStopWatch stopwatch,
            int minHfrIndex,
            IProgress<ApplicationStatus> progress) {
            int imagesAligned = 0;

            var targetStars = allDetectedStars[minHfrIndex].StarDetectionResult.StarList
                .Select(s => new Point2D(s.Position.X, s.Position.Y));
            double maxNormalisedBrightnessDiff = .1d;
            int maxDistanceForPutativeMatch = 25;
            for (int imageIndex = 0; imageIndex < allDetectedStars.Count; ++imageIndex) {
                if (imageIndex == minHfrIndex) {
                    continue;
                }
                ApplicationStatus status = new ApplicationStatus() {
                    Status = "Aligning images",
                    Status2 = "Image",
                    ProgressType2 = ApplicationStatus.StatusProgressType.ValueOfMaxValue,
                    MaxProgress2 = allDetectedStars.Count,
                    Progress2 = imageIndex + 1
                };
                progress.Report(status);
                var theseStars = allDetectedStars[imageIndex].StarDetectionResult.StarList
                    .Select(s => new Point2D(s.Position.X, s.Position.Y));
                var (putativeSrc, putativeDst) = RANSACRegistration.GeneratePutativeMatches(
                    theseStars.ToList(),
                    targetStars.ToList(),
                    maxDistanceForPutativeMatch, maxNormalisedBrightnessDiff,
                    null);
                try {
                    // calculate the transform needed to register this image
                    var transform = RANSACRegistration.EstimateSimilarityTransform(putativeSrc, putativeDst, status, progress);

                    // adjust each star according to the transform
                    for (int starIndex = 0; starIndex < allDetectedStars[imageIndex].StarDetectionResult.StarList.Count; starIndex++) {
                        var oldPoint = allDetectedStars[imageIndex].StarDetectionResult.StarList[starIndex].Position;
                        var transformedPoint = transform.Transform(new Point2D(allDetectedStars[imageIndex].StarDetectionResult.StarList[starIndex].Position));
                        allDetectedStars[imageIndex].StarDetectionResult.StarList[starIndex].Position = new Accord.Point((float)transformedPoint.X, (float)transformedPoint.Y);
                    }
                    imagesAligned++;
                } catch (Exception ex) {
                    Trace.WriteLine($"Image {imageIndex}: Error: {ex.Message}");
                }
            }

            stopwatch.RecordEntry("RANSAC alignment");
            return imagesAligned;
        }

        private static (RegisteredStar[], int) MatchStarsUsingKdTree(
                List<SensorDetectedStars> allDetectedStars,
                MultiStopWatch stopwatch,
                int minHfrIndex,
                float searchRadius,
                double maxNormalisedBrightnessDiff,
                IProgress<ApplicationStatus> progress) {
            RegisteredStar[] registeredStars;
            var allDetectedStarTrees = allDetectedStars.Select(result => {
                var tree = new KdTree<float, DetectedStarIndex>(2, new FloatMath(), AddDuplicateBehavior.Error);
                foreach (var (star, starIndex) in result.StarDetectionResult.StarList.Select((star, starIndex) => ((HocusFocusDetectedStar)star, starIndex))) {
                    tree.Add(new[] { star.Position.X, star.Position.Y }, new DetectedStarIndex(starIndex, star));
                }
                return tree;
            }).ToArray();
            stopwatch.RecordEntry("build trees");

            var globalRegistry = new KdTree<float, DetectedStarIndex>(2, new FloatMath(), AddDuplicateBehavior.Error);
            var starIndexMap = Enumerable.Range(0, allDetectedStars.Count).Select(i => new Dictionary<int, int>()).ToArray();
            foreach (var starNode in allDetectedStarTrees[minHfrIndex]) {
                var nextIndex = globalRegistry.Count;
                globalRegistry.Add(starNode.Point, new DetectedStarIndex(nextIndex, starNode.Value.DetectedStar));
                starIndexMap[minHfrIndex].Add(starNode.Value.Index, nextIndex);
            }

            ApplicationStatus status = new ApplicationStatus() {
                Status = "Matching stars",
                MaxProgress = allDetectedStars.Count,
                ProgressType = ApplicationStatus.StatusProgressType.ValueOfMaxValue
            };
            int totalRejectionsOnBrightness = 0;
            for (int imageIndex = 0; imageIndex < allDetectedStars.Count; ++imageIndex) {
                if (imageIndex == minHfrIndex) {
                    continue;
                }
                status.Progress = imageIndex;
                progress.Report(status);

                var nextStarList = allDetectedStars[imageIndex].StarDetectionResult.StarList;
                var nextStarTree = allDetectedStarTrees[imageIndex];
                var nextStarIndexMap = starIndexMap[imageIndex];
                var matchedGlobalStars = new bool[globalRegistry.Count];
                var matchedSourceStars = new bool[nextStarTree.Count];
                var queue = new KdTree.PriorityQueue<MatchingPair, double>(new DoubleMath());
                foreach (var (starNode, starNodeIndex) in nextStarTree.Select((starNode, starNodeIndex) => (starNode, starNodeIndex))) {
                    var sourceStar = starNode.Value.DetectedStar;
                    var sourcePoint = starNode.Point;
                    var sourceIndex = starNode.Value.Index;
                    var globalNeighbors = globalRegistry.RadialSearch(sourcePoint, searchRadius);
                    int queuedCount = 0;
                    foreach (var globalNeighbor in
                        maxNormalisedBrightnessDiff == -1 ? globalNeighbors :
                        // filter out bad matches on brightness
                        globalNeighbors.Where(p => Math.Abs(p.Value.DetectedStar.NormalisedBrightness - sourceStar.NormalisedBrightness) < maxNormalisedBrightnessDiff)
                        ) {
                        var globalNeighborIndex = globalNeighbor.Value.Index;
                        var neighboringStar = globalNeighbor.Value.DetectedStar;
                        var distance = MathUtility.DotProduct(globalNeighbor.Point, sourcePoint);
                        queue.Enqueue(new MatchingPair() { SourceIndex = sourceIndex, GlobalIndex = globalNeighborIndex }, -distance);
                        queuedCount++;
                    }
                    int rejectedOnBrightness = globalNeighbors.Count() - queuedCount;
                    if (rejectedOnBrightness > 0)
                        Trace.WriteLine($"ImageStar: {imageIndex}/{sourceIndex}, {rejectedOnBrightness} matches out of {globalNeighbors.Count()} rejected on brightness difference (>{maxNormalisedBrightnessDiff})");
                    totalRejectionsOnBrightness += rejectedOnBrightness;
                }

                //double totalDiff = 0d;
                //int neighborCount = 0;
                status.Status = "Measuring matched star offset";
                status.MaxProgress = queue.Count;
                status.ProgressType = ApplicationStatus.StatusProgressType.ValueOfMaxValue;
                while (queue.Count > 0) {
                    var nextCandidate = queue.Dequeue();
                    if (matchedGlobalStars[nextCandidate.GlobalIndex] || matchedSourceStars[nextCandidate.SourceIndex]) {
                        continue;
                    }

                    status.Progress = (status.MaxProgress - queue.Count) + 1;
                    progress.Report(status);

                    nextStarIndexMap.Add(nextCandidate.SourceIndex, nextCandidate.GlobalIndex);
                    matchedGlobalStars[nextCandidate.GlobalIndex] = true;
                    matchedSourceStars[nextCandidate.SourceIndex] = true;

                    /* next lines are just to evaluate benefit of ransac alignment
                    var dist = DistanceSquared(globalRegistry.ElementAt(nextCandidate.GlobalIndex).Point,
                        new float[] { allDetectedStars[imageIndex].StarDetectionResult.StarList[nextCandidate.SourceIndex].Position.X, allDetectedStars[imageIndex].StarDetectionResult.StarList[nextCandidate.SourceIndex].Position.Y });
                    totalDiff += dist;
                    neighborCount++;*/
                }
                //Trace.WriteLine($"Image {imageIndex}: average matching star distance: {totalDiff / (float)neighborCount}");
            }

            registeredStars = new RegisteredStar[globalRegistry.Count];
            foreach (var globalNode in globalRegistry) {
                var registeredStar = new RegisteredStar() {
                    RegistrationX = globalNode.Value.DetectedStar.Position.X,
                    RegistrationY = globalNode.Value.DetectedStar.Position.Y
                };
                registeredStars[globalNode.Value.Index] = registeredStar;
            }

            status.Status = "Registering matched stars";
            status.MaxProgress = starIndexMap.Length;

            for (int i = 0; i < starIndexMap.Length; ++i) {
                var nextStarIndexMap = starIndexMap[i];
                var focuserPosition = allDetectedStars[i].FocuserPosition;
                var detectedStars = allDetectedStars[i].StarDetectionResult.StarList;

                status.Progress = i;
                progress.Report(status);

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

            return (registeredStars, totalRejectionsOnBrightness);
        }

        private (RegisteredStar[], int) MatchStarsWithoutKdTree(
            List<SensorDetectedStars> allDetectedStars,
            System.Drawing.Size imageSize,
            MultiStopWatch stopwatch,
            int minHfrIndex,
            int searchSquareSide,
            double maxRelativeBrightnessDiff,
            IProgress<ApplicationStatus> progress) {
            List<StarDetectionRegion> regions = CreateFullRegionSet(imageSize, 2, 3);
            RegisteredStar[] registeredStars = new RegisteredStar[0];

            // create dictionary of stars in the reference image
            Dictionary<Point2D, List<MatchedStar>> starDict = new(allDetectedStars.Count * allDetectedStars[minHfrIndex].StarDetectionResult.StarList.Count);
            foreach (var star in allDetectedStars[minHfrIndex].StarDetectionResult.StarList
                .Select(star => (HocusFocusDetectedStar)star)) {
                starDict.Add(new Point2D(star.Position.X, star.Position.Y, star.NormalisedBrightness), new List<MatchedStar>() { new MatchedStar() {
                                FocuserPosition = allDetectedStars[minHfrIndex].FocuserPosition,
                                Star = (HocusFocusDetectedStar)star,
                                ImageIndex = minHfrIndex
                            } });
            }

            int allStarCount = starDict.Count;
            // remove the dimmest stars so we only keep the brightest ones
            int minMatchPct = 10;   // proportion of images that must have star for star to be included
            double minBrightness = starDict.Keys.OrderBy(s => s.NormalisedBrightness).ElementAt((int)(allStarCount * .05)).NormalisedBrightness; // min set allow 95% of stars
            Trace.WriteLine($"Minimum brightness set to {minBrightness}");

            int brightestStarCount = allStarCount;
            if (minBrightness != 0) {
                foreach (var pt in starDict.Keys.Where(pt => pt.NormalisedBrightness < minBrightness))
                    starDict.Remove(pt);
                brightestStarCount = starDict.Count;
            }
            int rejectionsOnBrightness = 0;

            stopwatch.RecordEntry("Dictionary");

            ApplicationStatus status = new ApplicationStatus() {
                Status = "Matching stars",
                Status2 = "Image",
                MaxProgress2 = allDetectedStars.Count,
                ProgressType2 = ApplicationStatus.StatusProgressType.ValueOfMaxValue,
            };
            // in other images find stars that are close to the reference image
            for (int i = 0; i < allDetectedStars.Count; ++i) {
                if (i == minHfrIndex) {
                    continue;
                }
                status.Progress2 = i;
                status.Status3 = "Star";
                status.MaxProgress3 = allDetectedStars[i].StarDetectionResult.StarList.Count;
                status.ProgressType3 = ApplicationStatus.StatusProgressType.ValueOfMaxValue;
                status.Progress3 = 0;
                foreach (var star in allDetectedStars[i].StarDetectionResult.StarList
                    .Select(star => (HocusFocusDetectedStar)star)) {
                    status.Progress3++;
                    progress.Report(status);

                    var searchPoint = new Point2D(star.Position.X, star.Position.Y, star.NormalisedBrightness);
                    var searchArea = new Rect2d(searchPoint.X - searchSquareSide, searchPoint.Y - searchSquareSide, searchSquareSide * 2 + 1, searchSquareSide * 2 + 1);
                    // find a star in the reference image that is within the search area and closest to the current star
                    var starKeysInArea = starDict.Keys
                        .Where(p => searchArea.Contains(p.X, p.Y));
                    var refStar =
                        (maxRelativeBrightnessDiff == -1 ?
                            starKeysInArea :
                            starKeysInArea.Where(p => (Math.Abs(star.NormalisedBrightness - p.NormalisedBrightness) < maxRelativeBrightnessDiff))
                        )
                        .OrderBy(p => DistanceSquared(p, searchPoint))
                        .FirstOrDefault();
                    if (maxRelativeBrightnessDiff != -1) {
                        rejectionsOnBrightness += starKeysInArea
                                .Where(p => (Math.Abs(star.NormalisedBrightness - p.NormalisedBrightness) >= maxRelativeBrightnessDiff)).Count();
                    }
                    if (refStar != null) {
                        starDict[refStar].Add(new MatchedStar() {
                            FocuserPosition = allDetectedStars[i].FocuserPosition,
                            Star = (HocusFocusDetectedStar)star,
                            ImageIndex = i
                        });
                    }
                }
            }

            stopwatch.RecordEntry("Match");

            int starCount = starDict.Count;
            int minStarsPerRegion = starCount / regions.Count / 4;  // minimum of 25% of detected stars must be matched
            Trace.WriteLine($"MinStarsPerRegion set to {minStarsPerRegion} ({starCount} total stars)");
            Dictionary<Rect2d, int> starsPerRegion = new Dictionary<Rect2d, int>(regions.Count); // item1=stars in the region that are matched, item2=stars in the region
            regions.ForEach(r => {
                Rect2d rect = r.OuterBoundary.ToRect2D(imageSize);
                if (!starsPerRegion.ContainsKey(rect))
                    starsPerRegion.Add(rect, 0);
            });

            foreach (var pt in starDict.Keys) {
                starsPerRegion.Keys
                                .Where(r => r.Contains(pt.X, pt.Y))
                                .ToList()
                                .ForEach(r => starsPerRegion[r]++);
            }
            status.Status = "Examining region";
            status.MaxProgress = regions.Count;
            status.ProgressType = ApplicationStatus.StatusProgressType.ValueOfMaxValue;
            status.Progress = 0;
            foreach (var region in regions) {
                status.Progress++;
                progress.Report(status);

                for (int matchPct = 100; matchPct > minMatchPct; matchPct -= 10) {
                    int matchCount = 0;
                    var registeredStarsInRegion = starDict.Keys
                        .Where(pt => (region.OuterBoundary.ToRect2D(imageSize).Contains(pt.X, pt.Y)) &&    // just this region
                                    (starDict[pt].Count >= allDetectedStars.Count * matchPct / 100))   // only use stars that are matched by matchPct% of the images
                        .Select(pt => {
                            var rs = new RegisteredStar() {
                                RegistrationX = pt.X,
                                RegistrationY = pt.Y,
                            };
                            foreach (var focuserPositionGroup in starDict[pt].GroupBy(s => s.FocuserPosition)) {
                                var focuserPosition = focuserPositionGroup.Key;
                                var firstStarIngroup = focuserPositionGroup.First().Star;
                                MatchedStar aveStar = new MatchedStar() {
                                    FocuserPosition = focuserPosition,
                                    Star = new() {
                                        BoundingBox = firstStarIngroup.BoundingBox,
                                        Position = firstStarIngroup.Position,
                                        PSF = firstStarIngroup.PSF,
                                        Background = focuserPositionGroup.Average(g => g.Star.Background),
                                        AverageBrightness = focuserPositionGroup.Average(g => g.Star.AverageBrightness),
                                        HFR = focuserPositionGroup.Average(g => g.Star.HFR),
                                        MaxBrightness = focuserPositionGroup.Average(g => g.Star.MaxBrightness),
                                    }
                                };
                                rs.MatchedStars.Add(aveStar);
                            }
                            matchCount++;
                            return rs;
                        }).ToArray();

                    registeredStars = registeredStars.Concat(registeredStarsInRegion).ToArray();
                    if (matchCount > minStarsPerRegion) { // enough stars so stop now
                        Trace.WriteLine($"Region {region.Index}: {matchCount} stars matched with with matchPct set to {matchPct}");
                        break;
                    }
                    Trace.WriteLine($"Region {region.Index}: Too few stars ({matchCount}) with matchPct set to {matchPct}");
                }
            }

            stopwatch.RecordEntry("NonTree-based registration");

            if (minBrightness > 0) {
                Trace.WriteLine($"Of all {allStarCount} stars found, {brightestStarCount} stars were selected on brightness (> {minBrightness})");
            }
            Trace.WriteLine($"{starCount} stars were registered and matched to stars in other images.  Stars on enough images are {registeredStars.Length}");
            return (registeredStars, rejectionsOnBrightness);
        }

        private static double DistanceSquared(Point2D p1, Point2D p2) {
            return DistanceSquared(p1.X, p1.Y, p2.X, p2.Y);
        }

        private static int DistanceSquared(int x1, int y1, int x2, int y2) {
            int xDist = x1 - x2;
            int yDist = y1 - y2;
            return (xDist * xDist + yDist * yDist);
        }

        private static double DistanceSquared(double x1, double y1, double x2, double y2) {
            double xDist = x1 - x2;
            double yDist = y1 - y2;
            return (xDist * xDist + yDist * yDist);
        }

        private static double DistanceSquared(double[] p1, double[] p2) {
            double ret = 0d;
            for (int i = 0; i < p1.Length; i++) {
                double dist = p1[i] - p2[i];
                ret += dist * dist;
            }

            return ret;
        }

        private static double DistanceSquared(float[] p1, float[] p2) {
            float ret = 0f;
            for (int i = 0; i < p1.Length; i++) {
                float dist = p1[i] - p2[i];
                ret += dist * dist;
            }

            return ret;
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