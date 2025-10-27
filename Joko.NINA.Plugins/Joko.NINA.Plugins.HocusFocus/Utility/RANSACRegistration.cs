#region "copyright"

/*
    Copyright © 2021 - 2021 George Hilios <ghilios+NINA@googlemail.com>

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#endregion "copyright"

using MathNet.Numerics.LinearAlgebra;
using NINA.Core.Model;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

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

        public Point2D(Accord.Point point) {
            X = point.X;
            Y = point.Y;
            NormalisedBrightness = 0;
        }

        public PointF AsPointF() {
            return new PointF((float)X, (float)Y);
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

    public class RANSACRegistration {
        private static Random RNG = new();

        // Generate putative matches using nearest neighbor
        public static (List<Point2D> srcPoints, List<Point2D> dstPoints) GeneratePutativeMatchesUsingNN(
            List<Point2D> imageStars,
            List<Point2D> referenceStars,
            double maxDistance,
            double relativeBrightnessDiff,
            ApplicationStatus status) {
            var srcPoints = new List<Point2D>();
            var dstPoints = new List<Point2D>();
            double maxDistance2 = maxDistance * maxDistance;    // distance stays as squared

            if (status != null) {
                status.Status3 = "Generating putative matches";
                status.ProgressType3 = ApplicationStatus.StatusProgressType.ValueOfMaxValue;
                status.MaxProgress3 = imageStars.Count * imageStars.Count;
                status.Progress3 = 0;
            }
            // For each star in this image, find closest match in the reference image
            foreach (var imageStar in imageStars) {
                Point2D bestMatch = null;
                double bestDistance = double.MaxValue;

                foreach (var referenceStar in referenceStars) {
                    if (status != null) {
                        status.Progress3++;
                    }
                    if (Math.Abs(imageStar.NormalisedBrightness - referenceStar.NormalisedBrightness) <= relativeBrightnessDiff) {
                        double x = imageStar.X - referenceStar.X;
                        double y = imageStar.Y - referenceStar.Y;
                        double distance2 = x * x + y * y;

                        if (distance2 < bestDistance && distance2 < maxDistance2) {
                            bestDistance = distance2;
                            bestMatch = referenceStar;
                        }
                    }
                }

                if (bestMatch != null) {
                    srcPoints.Add(imageStar);
                    dstPoints.Add(bestMatch);
                }
            }

            return (srcPoints, dstPoints);
        }

        public class StarTriangle {
            private int referenceID;
            private List<double> normalizedLengths;
            private List<double> normalizedBrightnesses;
            private bool matched;
            private bool isReference;   // just used for annotating images with triangles
            private double matchScore;

            public Point2D P1 { get; }
            public Point2D P2 { get; }
            public Point2D P3 { get; }

            public StarTriangle(Point2D p1, Point2D p2, Point2D p3, bool isReference, int referenceID) {
                P1 = p1;
                P2 = p2;
                P3 = p3;
                this.isReference = isReference;
                if (isReference)
                    this.referenceID = referenceID;

                normalizedLengths = normalizeLength();
                normalizedBrightnesses = normalizeBrightnesses();
            }

            private List<double> normalizeLength() {
                var lengths = new List<double> { lineLength(P1, P2), lineLength(P2, P3), lineLength(P3, P1) }.Order().ToList();
                double shortestLength = lengths.First();
                lengths.ForEach(l => l = l / shortestLength);
                return lengths;
            }

            private List<double> normalizeBrightnesses() {
                var brightnesses = new List<double> { P1.NormalisedBrightness, P2.NormalisedBrightness, P3.NormalisedBrightness }.Order().ToList();
                double dimmest = brightnesses.First();
                brightnesses.ForEach(b => b = b / dimmest);
                return brightnesses;
            }

            private double lineLength(Point2D p1, Point2D p2) {
                double x = p2.X - p1.X;
                double y = p2.Y - p1.Y;
                return x * x + y * y;
            }

            public List<double> NormalizedLengths { get => normalizedLengths; }
            public List<double> NormalizedBrightnesses { get => normalizedBrightnesses; }

            public bool IsMatchOnLength(StarTriangle other, double tolerance) {
                for (int i = 0; i < this.NormalizedLengths.Count; i++)
                    if (Math.Abs(this.NormalizedLengths[0] - other.NormalizedLengths[0]) > tolerance)
                        return false;
                return true;
            }

            public bool IsMatchOnBrightness(StarTriangle other, double tolerance) {
                for (int i = 0; i < this.NormalizedBrightnesses.Count; i++)
                    if (Math.Abs(this.NormalizedBrightnesses[0] - other.NormalizedBrightnesses[0]) > tolerance)
                        return false;
                return true;
            }

            public bool IsMatch(StarTriangle other, double brightnessTolerance, double lengthTolerance) {
                return IsMatchOnLength(other, lengthTolerance) && IsMatchOnBrightness(other, brightnessTolerance);
            }

            public double[] AsVector() {
                return new double[] {
                    NormalizedLengths[0], NormalizedLengths[1], NormalizedLengths[2],
                    NormalizedBrightnesses[0], NormalizedBrightnesses[1], NormalizedBrightnesses[2]
                };
            }

            public bool Matched { get => matched; }
            public bool IsReference { get => isReference; }
            public int ReferenceID { get => referenceID; }

            public override string ToString() {
                return $"StarTriangle: ({P1.X}, {P1.Y}), ({P2.X}, {P2.Y}), ({P3.X}, {P3.Y})";
            }

            public void MarkAsMatched(int referenceID, double matchScore) {
                matched = true;
                this.referenceID = referenceID;
                this.matchScore = matchScore;
            }

            public string MatchString {
                get { return $"{ReferenceID} ({matchScore:0.#########})"; }
            }
        }

        public static List<StarTriangle> BuildStarTriangles(List<Point2D> point2Ds, int searchSquareSide, bool onePerPoint, bool isReference) {
            // first pass - build list of triangles in reference image
            var triangles = new List<StarTriangle>();
            var pointsUsed = new HashSet<Point2D>();
            int id = 0;
            for (int i = 0; i < point2Ds.Count; i++) {
                if (pointsUsed.Contains(point2Ds[i]))
                    continue;
                var pt = point2Ds[i];
                var searchArea = new Rect2d(pt.X - searchSquareSide, pt.Y - searchSquareSide, searchSquareSide * 2, searchSquareSide * 2);
                var nearbyPoints = point2Ds
                    .Where(p => !pointsUsed.Contains(p) && searchArea.Contains(p.X, p.Y) && (p != pt))
                    .OrderBy(p => (p.X * p.X + p.Y * p.Y) - (pt.X * pt.X + pt.Y * pt.Y))
                    .ToList();
                if (onePerPoint) {
                    if (nearbyPoints.Count > 2) {
                        triangles.Add(new StarTriangle(pt, nearbyPoints[0], nearbyPoints[1], isReference, id++));
                        pointsUsed.Add(pt);
                        pointsUsed.Add(nearbyPoints[0]);
                        pointsUsed.Add(nearbyPoints[1]);
                    }
                } else {
                    for (int j = 0; j < nearbyPoints.Count; j++) {
                        var p2 = nearbyPoints[j];
                        for (int k = j + 1; k < nearbyPoints.Count; k++) {
                            var p3 = nearbyPoints[k];
                            var triangle = new StarTriangle(pt, p2, p3, isReference, id++);
                            triangles.Add(triangle);
                            //pointsUsed.Add(pt);
                            //pointsUsed.Add(p2);
                            //pointsUsed.Add(p3);
                        }
                    }
                }
            }
            return triangles;
        }

        // Generate putative matches using triangles
        public static (List<Point2D> srcPoints, List<Point2D> dstPoints) GeneratePutativeMatchesUsingSimilarTriangles(
            List<StarTriangle> imageTriangles,
            List<StarTriangle> referenceTriangles,
            ApplicationStatus status) {
            var srcPoints = new List<Point2D>();
            var dstPoints = new List<Point2D>();

            if (status != null) {
                status.Status3 = "Generating putative matches (robust)";
                status.ProgressType3 = ApplicationStatus.StatusProgressType.ValueOfMaxValue;
                status.MaxProgress3 = imageTriangles.Count * referenceTriangles.Count;
                status.Progress3 = 0;
            }

            // For each triangle in the reference image, find closest match in this image
            double minCosSim = 0.99; // Cosine similarity threshold for accepting a match

            foreach (var referenceTriangle in referenceTriangles) {
                if (status != null) {
                    status.Progress3++;
                }
                double bestCosSim = minCosSim;
                StarTriangle bestMatch = null;
                foreach (var imageTriangle in imageTriangles.Where(t => !t.Matched)) {
                    var cosSim = CosineSimilarity(referenceTriangle.AsVector(), imageTriangle.AsVector());

                    if (cosSim > bestCosSim) {
                        bestCosSim = cosSim;
                        bestMatch = imageTriangle;
                    }
                }
                if ((bestMatch != null) && (bestCosSim > minCosSim)) {
                    srcPoints.AddRange(new List<Point2D> { bestMatch.P1, bestMatch.P2, bestMatch.P3 });
                    dstPoints.AddRange(new List<Point2D> { referenceTriangle.P1, referenceTriangle.P2, referenceTriangle.P3 });
                    bestMatch.MarkAsMatched(referenceTriangle.ReferenceID, bestCosSim);
                }
            }

            return (srcPoints, dstPoints);
        }

        private static double CosineSimilarity(double[] vecA, double[] vecB) {
            double dotProduct = 0;
            double magnitudeA = 0;
            double magnitudeB = 0;
            for (int i = 0; i < vecA.Length; i++) {
                dotProduct += vecA[i] * vecB[i];
                magnitudeA += vecA[i] * vecA[i];
                magnitudeB += vecB[i] * vecB[i];
            }
            if (magnitudeA == 0 || magnitudeB == 0) {
                return 0;
            }
            return dotProduct / (Math.Sqrt(magnitudeA) * Math.Sqrt(magnitudeB));
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