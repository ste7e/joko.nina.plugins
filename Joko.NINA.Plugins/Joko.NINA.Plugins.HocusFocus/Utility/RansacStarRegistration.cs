#region "copyright"

/*
    Copyright © 2021 - 2021 George Hilios <ghilios+NINA@googlemail.com>

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#endregion "copyright"

using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NINA.Joko.Plugins.HocusFocus.Utility {

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

    public class RansacStarRegistration {

        public static SimilarityTransform EstimateSimilarityTransform(
            List<Point2D> srcPoints,
            List<Point2D> dstPoints,
            int maxIterations = 1000,
            double inlierThreshold = 5.0,
            int minInliers = 4) {
            if (/*srcPoints.Count != dstPoints.Count ||*/ srcPoints.Count < 2) {
                throw new ArgumentException("At least 2 corresponding points are required.");
            }

            Random rand = new Random();
            int bestInlierCount = 0;
            List<int> bestInlierIndices = new List<int>();
            SimilarityTransform bestTransform = null;

            for (int i = 0; i < maxIterations; i++) {
                // Randomly select 2 points for similarity transform
                int[] indices = Enumerable.Range(0, Math.Min(srcPoints.Count, dstPoints.Count)).OrderBy(x => rand.Next()).Take(2).ToArray();
                var sampleSrc = indices.Select(idx => srcPoints[idx]).ToList();
                var sampleDst = indices.Select(idx => dstPoints[idx]).ToList();

                // Estimate transformation from sample
                var transform = FitSimilarityTransform(sampleSrc, sampleDst);
                if (transform == null) continue;

                // Count inliers
                List<int> inlierIndices = new List<int>();
                for (int j = 0; j < Math.Min(srcPoints.Count, dstPoints.Count); j++) {
                    var transformedPoint = transform.Transform(srcPoints[j]);
                    double distance = Math.Sqrt(
                        Math.Pow(transformedPoint.X - dstPoints[j].X, 2) +
                        Math.Pow(transformedPoint.Y - dstPoints[j].Y, 2));
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

        /*        // Example usage
                public static void Main() {
                    // Example: 5 images with star positions and correspondences
                    var referenceStars = new List<Point2D>
                    {
                        new Point2D(100, 100), new Point2D(200, 150), new Point2D(150, 200),
                        new Point2D(300, 250), new Point2D(120, 180)
                    };

                    var targetStars = new List<Point2D>
                    {
                        new Point2D(110, 105), new Point2D(210, 160), new Point2D(160, 210),
                        new Point2D(310, 260), new Point2D(130, 190) // Simulated slight offset
                    };

                    try {
                        var transform = EstimateSimilarityTransform(referenceStars, targetStars);
                        Console.WriteLine($"Scale: {transform.Scale:F4}");
                        Console.WriteLine($"Rotation: {transform.Rotation:F4} radians");
                        Console.WriteLine($"Translation: ({transform.Tx:F4}, {transform.Ty:F4})");

                        // Transform a point as a test
                        var testPoint = new Point2D(100, 100);
                        var transformedPoint = transform.Transform(testPoint);
                        Console.WriteLine($"Transformed point: ({transformedPoint.X:F4}, {transformedPoint.Y:F4})");
                    } catch (Exception ex) {
                        Console.WriteLine($"Error: {ex.Message}");
                    }
                }
        */
    }
}