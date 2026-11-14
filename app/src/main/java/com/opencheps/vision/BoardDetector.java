
package com.opencheps.vision;

import android.graphics.Bitmap;
import android.util.Log;

public class BoardDetector {

    public static class Board {
        public final int left, top, side;
        public final float square;
        public final double confidence;

        public Board(int left, int top, int side, double confidence) {
            this.left = left;
            this.top = top;
            this.side = side;
            this.square = side / 8f;
            this.confidence = confidence;
        }
    }

    private static final String TAG = "OC/Detector";
    private static final int WORK_MAX_EDGE = 480;
    private static final double ACCEPT_THRESHOLD = 18.0;

    public static Board locate(Bitmap src) {
        if (src == null) return null;

        long t0 = System.nanoTime();

        int W0 = src.getWidth();
        int H0 = src.getHeight();

        float scale = Math.min(
                1f, (float) WORK_MAX_EDGE / Math.max(W0, H0));

        int W = Math.max(64, Math.round(W0 * scale));
        int H = Math.max(64, Math.round(H0 * scale));

        Bitmap work = (scale < 0.999f)
                ? Bitmap.createScaledBitmap(src, W, H, true)
                : src;

        int[] argb = new int[W * H];
        work.getPixels(argb, 0, W, 0, 0, W, H);

        if (work != src) work.recycle();

        int[] gray = new int[W * H];

        for (int i = 0; i < gray.length; i++) {
            int p = argb[i];
            int r = (p >> 16) & 0xFF;
            int g = (p >> 8) & 0xFF;
            int b = p & 0xFF;
            gray[i] = (r * 299 + g * 587 + b * 114) / 1000;
        }

        int Wp = W + 1;
        long[] ii = new long[Wp * (H + 1)];

        for (int y = 0; y < H; y++) {
            long row = 0;
            int srcBase = y * W;
            int dstBase = (y + 1) * Wp;
            int prevBase = y * Wp;

            for (int x = 0; x < W; x++) {
                row += gray[srcBase + x];
                ii[dstBase + x + 1] =
                        ii[prevBase + x + 1] + row;
            }
        }

        int minSide = Math.max(80, Math.min(W, H) / 3);
        int maxSide = Math.min(W, H);

        Candidate best = null;

        int sizeStep = Math.max(2, (maxSide - minSide) / 36);

        for (int s = minSide; s <= maxSide; s += sizeStep) {
            int cell = s / 8;
            if (cell < 5) continue;

            int corrected = cell * 8;
            int posStep = Math.max(2, cell / 3);

            for (int y = 0; y + corrected <= H; y += posStep) {
                for (int x = 0; x + corrected <= W; x += posStep) {
                    double score =
                            checkerScore(ii, Wp, x, y, cell);

                    if (best == null || score > best.score) {
                        best = new Candidate(
                                x, y, corrected, cell, score);
                    }
                }
            }
        }

        if (best == null) return null;

        Candidate refined = refine(ii, Wp, W, H, best);

        long elapsed =
                (System.nanoTime() - t0) / 1_000_000;

        Log.d(TAG, "locate: " + W + "x" + H
                + " best=" + refined.score
                + " bbox=(" + refined.x + "," + refined.y
                + ",s=" + refined.side + ")"
                + " in " + elapsed + " ms");

        if (refined.score < ACCEPT_THRESHOLD) return null;

        int left = Math.round(refined.x / scale);
        int top = Math.round(refined.y / scale);
        int side = Math.round(refined.side / scale);

        return new Board(left, top, side, refined.score);
    }

    private static Candidate refine(
            long[] ii, int Wp, int W, int H, Candidate c) {

        Candidate best = c;
        int cell = c.cell;

        for (int dy = -cell; dy <= cell; dy += 1) {
            for (int dx = -cell; dx <= cell; dx += 1) {
                int x = c.x + dx;
                int y = c.y + dy;

                if (x < 0 || y < 0
                        || x + c.side > W
                        || y + c.side > H) {
                    continue;
                }

                double score =
                        checkerScore(ii, Wp, x, y, cell);

                if (score > best.score) {
                    best = new Candidate(
                            x, y, c.side, cell, score);
                }
            }
        }

        return best;
    }

    private static double checkerScore(
            long[] ii, int Wp, int x0, int y0, int cell) {

        int samp = Math.max(2, cell / 5);
        int inset = Math.max(2, cell / 6);

        double[] light = new double[32];
        double[] dark = new double[32];

        int li = 0;
        int di = 0;

        double[] corners = new double[4];

        for (int r = 0; r < 8; r++) {
            for (int f = 0; f < 8; f++) {
                int cx = x0 + f * cell;
                int cy = y0 + r * cell;

                corners[0] = boxMean(ii, Wp,
                        cx + inset, cy + inset, samp);

                corners[1] = boxMean(ii, Wp,
                        cx + cell - inset - samp,
                        cy + inset, samp);

                corners[2] = boxMean(ii, Wp,
                        cx + inset,
                        cy + cell - inset - samp, samp);

                corners[3] = boxMean(ii, Wp,
                        cx + cell - inset - samp,
                        cy + cell - inset - samp, samp);

                sort4(corners);

                double median =
                        (corners[1] + corners[2]) * 0.5;

                if (((r + f) & 1) == 0) {
                    light[li++] = median;
                } else {
                    dark[di++] = median;
                }
            }
        }

        double meanL = mean(light);
        double meanD = mean(dark);

        double sigmaL = std(light, meanL);
        double sigmaD = std(dark, meanD);

        double separation = Math.abs(meanL - meanD);

        return separation * separation
                / (4.0 + sigmaL + sigmaD);
    }

    private static void sort4(double[] a) {
        double t;

        if (a[0] > a[1]) {
            t = a[0]; a[0] = a[1]; a[1] = t;
        }
        if (a[2] > a[3]) {
            t = a[2]; a[2] = a[3]; a[3] = t;
        }
        if (a[0] > a[2]) {
            t = a[0]; a[0] = a[2]; a[2] = t;
        }
        if (a[1] > a[3]) {
            t = a[1]; a[1] = a[3]; a[3] = t;
        }
        if (a[1] > a[2]) {
            t = a[1]; a[1] = a[2]; a[2] = t;
        }
    }

    private static double mean(double[] a) {
        double sum = 0;
        for (double v : a) sum += v;
        return sum / a.length;
    }

    private static double std(double[] a, double m) {
        double sum = 0;

        for (double v : a) {
            double d = v - m;
            sum += d * d;
        }

        return Math.sqrt(sum / a.length);
    }

    private static double boxMean(
            long[] ii, int Wp, int x, int y, int s) {

        int x2 = x + s;
        int y2 = y + s;

        long sum = ii[y2 * Wp + x2]
                - ii[y * Wp + x2]
                - ii[y2 * Wp + x]
                + ii[y * Wp + x];

        return (double) sum / (s * s);
    }

    private static final class Candidate {
        final int x, y, side, cell;
        final double score;

        Candidate(int x, int y, int side,
                int cell, double score) {
            this.x = x;
            this.y = y;
            this.side = side;
            this.cell = cell;
            this.score = score;
        }
    }
}
