// Example: RANSAC for line fitting in C#
using Accord.Imaging.Filters;
using MathNet.Numerics;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

public class RansacLineFitting
{
/*    public static void RansacTest()
    {
        // Generate sample data: y = 2x + 1 with some outliers
        var points = new List<(double x, double y)>();
        var rand = new Random();
        for (int i = 0; i < 50; i++)
        {
            double x = i;
            double y = 2 * x + 1 + rand.NextDouble() * 2 - 1; // small noise
            points.Add((x, y));
        }
        // Add outliers
        points.Add((10, 100));
        points.Add((20, -50));
        points.Add((30, 80));

        // Run RANSAC
        var (bestSlope, bestIntercept) = Ransac(points, 100, 1.5, 30);

        Console.WriteLine($"Best fit: y = {bestSlope:F2}x + {bestIntercept:F2}");
    }
*/
    class pair : IComparable { public (float x, float y)[] points;

        public int CompareTo(object obj)
        {
            if (obj is pair other)
            {
                int ret;
                ret = this.points[0].x.CompareTo(other.points[0].x);
                if (ret == 0)
                {
                    ret = this.points[0].y.CompareTo(other.points[0].y);
                    if (ret == 0)
                    {
                        ret = this.points[1].x.CompareTo(other.points[1].x);
                        if (ret == 0)
                        {
                            ret = this.points[1].y.CompareTo(other.points[1].y);
                        }
                    }
                }
                return ret;
            }
            throw new NotImplementedException();
        }
    }
    // RANSAC algorithm for line fitting
    public static (List<int>, List<double>) Ransac(
        List<(float x, float y)> points,
        double threshold,
        int minInliers)
    {
        int bestInlierCount = 0;
        double bestSlope = 0, bestIntercept = 0;
        int bestIndex = -1;

        // This is unRANSAC - i.e. non-random.  As the dataset is small all possible combinations of pairs will be considered for the line
        List<pair> pairs = new();
        for (int p1 = 0; p1 < points.Count; p1++)
            for (int p2 = p1 + 1; p2 < points.Count; p2++)
                pairs.Add(new() { points = new (float x, float y)[2] { points[p1], points[p2] } } );

        pairs.Sort();

        foreach (var pair in pairs)
        {
            var sample = pair.points;
            // Fit line: y = mx + b
            double m = (sample[1].y - sample[0].y) / ((sample[1].x == sample[0].x) ? 1 : (sample[1].x - sample[0].x));
            double b = sample[0].y - m * sample[0].x;

            // Count inliers with this sample pair
            int inliers = points.Count(p => Math.Abs(p.y - (m * p.x + b)) < threshold);

            if (inliers > bestInlierCount && inliers >= minInliers)
            {
                bestInlierCount = inliers;
                bestSlope = m;
                bestIntercept = b;
            }
/*            if (inliers > points.Count() * .8)
            {
                Trace.WriteLine($"RANSAC 80% inliers hit after {i} iterations");
                break;
            }
*/        }

        Trace.WriteLine($"RANSAC finished after {pairs.Count} iterations with {bestInlierCount} inliers {bestInlierCount * 100 / points.Count}% slope:{bestSlope}, intercept:{bestIntercept}");
        // load inliers
        var inlierIndices = new List<int>();
        var inlierFit = new List<double>();
        for (int index = 0; index < points.Count; index++)
        {
            var p = points[index];
            double fit = Math.Abs(p.y - (bestSlope * p.x + bestIntercept));
            if (fit < threshold)
            {
                inlierIndices.Add(index);
                inlierFit.Add(fit);
            }
        }

        return (inlierIndices, inlierFit);
    }

}