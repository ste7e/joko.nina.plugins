#region "copyright"

/*
    Copyright © 2021 - 2021 George Hilios <ghilios+NINA@googlemail.com>

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#endregion "copyright"

using Accord;
using MathNet.Numerics.LinearAlgebra;
using NINA.Core.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NINA.Joko.Plugins.HocusFocus.Utility {

    public class Point2D {
        public double X { get; private set; }
        public double Y { get; private set; }
        public double NormalisedBrightness { get; private set; } // Optional: for brightness filtering

        public Point2D(double x, double y, double normalisedBrightness = 0) {
            X = x;
            Y = y;
            NormalisedBrightness = normalisedBrightness;
        }

        public Point2D(Point point) {
            X = point.X;
            Y = point.Y;
            NormalisedBrightness = 0;
        }
    }

    public class SimilarityTransform {
        public double Scale { get; private set; }
        public double Rotation { get; private set; } // In radians
        public double Tx { get; private set; } // Translation X
        public double Ty { get; private set; } // Translation Y

        public SimilarityTransform(double scale, double rotation, double tx, double ty) {
            Scale = scale;
            Rotation = rotation;
            Tx = tx;
            Ty = ty;
        }

        public Point2D Transform(Point2D point) {
            double cosTheta = Math.Cos(Rotation);
            double sinTheta = Math.Sin(Rotation);
            double x = Scale * (cosTheta * point.X - sinTheta * point.Y) + Tx;
            double y = Scale * (sinTheta * point.X + cosTheta * point.Y) + Ty;
            return new Point2D(x, y);
        }
    }

    internal class RANSACRegistration {
        private static Random RNG = new();

        // Generate putative matches using nearest neighbor
        public static (List<Point2D> srcPoints, List<Point2D> dstPoints) GeneratePutativeMatches(
            List<Point2D> refStars,
            List<Point2D> targetStars,
            double maxDistance,
            double relativeBrightnessDiff,
            ApplicationStatus status) {
            var srcPoints = new List<Point2D>();
            var dstPoints = new List<Point2D>();
            double maxDistance2 = maxDistance * maxDistance;    // distance stays as squared

            if (status != null) {
                status.Status3 = "Generating putative matches";
                status.ProgressType3 = ApplicationStatus.StatusProgressType.ValueOfMaxValue;
                status.MaxProgress3 = refStars.Count * refStars.Count;
                status.Progress3 = 0;
            }
            // For each star in reference image, find closest in target image
            foreach (var refStar in refStars) {
                Point2D bestMatch = null;
                double bestDistance = double.MaxValue;

                foreach (var targetStar in targetStars) {
                    if (status != null) {
                        status.Progress++;
                    }
                    if (Math.Abs(refStar.NormalisedBrightness - targetStar.NormalisedBrightness) <= relativeBrightnessDiff) {
                        double x = refStar.X - targetStar.X;
                        double y = refStar.Y - targetStar.Y;
                        double distance2 = x * x + y * y;

                        if (distance2 < bestDistance && distance2 < maxDistance2) {
                            bestDistance = distance2;
                            bestMatch = targetStar;
                        }
                    }
                }

                if (bestMatch != null) {
                    srcPoints.Add(refStar);
                    dstPoints.Add(bestMatch);
                }
            }

            return (srcPoints, dstPoints);
        }

        public static SimilarityTransform EstimateSimilarityTransform(
            List<Point2D> srcPoints,
            List<Point2D> dstPoints,
            ApplicationStatus status,
            IProgress<ApplicationStatus> progress,
            int maxIterations = 1000,
            double inlierThreshold = 25,    // 5.0^2
            int minInliers = 4) {
            if (srcPoints.Count != dstPoints.Count || srcPoints.Count < 2) {
                throw new ArgumentException("At least 2 corresponding points are required.");
            }

            int bestInlierCount = 0;
            List<int> bestInlierIndices = new List<int>();
            SimilarityTransform bestTransform = null;

            status.Status3 = "Transform iteration";
            status.ProgressType3 = ApplicationStatus.StatusProgressType.ValueOfMaxValue;
            status.MaxProgress3 = maxIterations;
            for (int i = 0; i < maxIterations; i++) {
                status.Progress3 = i;
                progress.Report(status);
                
                // Randomly select 2 points for similarity transform
                int[] indices = Enumerable.Range(0, srcPoints.Count).OrderBy(x => RNG.Next()).Take(2).ToArray();
                var sampleSrc = indices.Select(idx => srcPoints[idx]).ToList();
                var sampleDst = indices.Select(idx => dstPoints[idx]).ToList();

                // Estimate transformation from sample
                var transform = FitSimilarityTransform(sampleSrc, sampleDst);
                if (transform == null) continue;

                // Count inliers
                List<int> inlierIndices = new List<int>();
                for (int j = 0; j < srcPoints.Count; j++) {
                    var transformedPoint = transform.Transform(srcPoints[j]);
                    double distX = transformedPoint.X - dstPoints[j].X;
                    double distY = transformedPoint.Y - dstPoints[j].Y;
                    double distance = distX * distX + distY * distY;
                    if (distance < inlierThreshold) {
                        inlierIndices.Add(j);
                    }
                }

                // Update best model if more inliers
                if (inlierIndices.Count > bestInlierCount && inlierIndices.Count >= minInliers) {
                    bestInlierCount = inlierIndices.Count;
                    bestInlierIndices = inlierIndices;
                }
            }

            // Refine transformation using all inliers
            if (bestInlierCount >= minInliers) {
                var inlierSrc = bestInlierIndices.Select(idx => srcPoints[idx]).ToList();
                var inlierDst = bestInlierIndices.Select(idx => dstPoints[idx]).ToList();
                bestTransform = FitSimilarityTransform(inlierSrc, inlierDst);
            } else {
                throw new InvalidOperationException("Not enough inliers to compute transformation.");
            }

            return bestTransform;
        }

        private static SimilarityTransform FitSimilarityTransform(List<Point2D> srcPoints, List<Point2D> dstPoints) {
            int n = srcPoints.Count;
            if (n < 2) return null;

            // Build design matrix A and vector b for least-squares: Ax = b
            var A = Matrix<double>.Build.Dense(2 * n, 4);
            var b = Vector<double>.Build.Dense(2 * n);

            for (int i = 0; i < n; i++) {
                double srcX = srcPoints[i].X;
                double srcY = srcPoints[i].Y;
                A[2 * i, 0] = srcX;
                A[2 * i, 1] = -srcY;
                A[2 * i, 2] = 1;
                A[2 * i, 3] = 0;
                A[2 * i + 1, 0] = srcY;
                A[2 * i + 1, 1] = srcX;
                A[2 * i + 1, 2] = 0;
                A[2 * i + 1, 3] = 1;

                b[2 * i] = dstPoints[i].X;
                b[2 * i + 1] = dstPoints[i].Y;
            }

            // Solve least-squares problem
            var x = A.Solve(b);
            if (x == null) return null;

            // Extract parameters: [s*cos(theta), s*sin(theta), tx, ty]
            double sCosTheta = x[0];
            double sSinTheta = x[1];
            double scale = Math.Sqrt(sCosTheta * sCosTheta + sSinTheta * sSinTheta);
            double rotation = Math.Atan2(sSinTheta, sCosTheta);
            double tx = x[2];
            double ty = x[3];

            return new SimilarityTransform(scale, rotation, tx, ty);
        }
    }
}