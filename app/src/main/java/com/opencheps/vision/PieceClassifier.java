package com.opencheps.vision;

import android.graphics.Bitmap;
import android.util.Log;

import com.opencheps.util.Side;

import java.util.Arrays;

public class PieceClassifier {

    private static final String TAG = "OC/Classifier";
    private static final int EMPTY_RANK_FIRST = 2;
    private static final int EMPTY_RANK_LAST = 5;

    private static final char[] START_WHITE_POV = {
        'r','n','b','q','k','b','n','r',
        'p','p','p','p','p','p','p','p',
        '.','.','.','.','.','.','.','.',
        '.','.','.','.','.','.','.','.',
        '.','.','.','.','.','.','.','.',
        '.','.','.','.','.','.','.','.',
        'P','P','P','P','P','P','P','P',
        'R','N','B','Q','K','B','N','R'
    };

    private SquareSampler.Sample[] templates;
    private SquareSampler.Sample[] previousFrame;
    private char[] previousLabels;
    private char[] labels;
    private float[][] emptyFeaturesByParity;
    private float emptyDistanceThreshold;
    private float frameChangeThreshold;
    private float whitePieceLumAvg;
    private float blackPieceLumAvg;
    private boolean trained;
    private Side bootstrapSide;

    public boolean isTrained() { return trained; }
    public Side bootstrapSide() { return bootstrapSide; }

    public synchronized boolean tryBootstrap(
            Bitmap bmp, BoardDetector.Board board) {
        if (bmp == null || board == null) return false;

        SquareSampler.Sample[] samples = sampleAll(bmp, board);
        if (samples == null) return false;

        float topLum = 0f;
        float bottomLum = 0f;
        for (int file = 0; file < 8; file++) {
            topLum += samples[file].meanLum
                    + samples[8 + file].meanLum;
            bottomLum += samples[48 + file].meanLum
                    + samples[56 + file].meanLum;
        }

        Side side = bottomLum > topLum
                ? Side.WHITE : Side.BLACK;

        char[] startingLabels = side == Side.WHITE
                ? Arrays.copyOf(START_WHITE_POV, 64)
                : flipLabels(START_WHITE_POV);

        float[][] emptyReferences = buildEmptyReferences(samples);
        if (emptyReferences == null) return false;

        float[] emptyDistances = new float[32];
        float[] occupiedDistances = new float[32];
        int emptyCount = 0;
        int occupiedCount = 0;

        for (int index = 0; index < 64; index++) {
            int rank = index / 8;
            int parity = squareParity(index);

            float distance = SquareSampler.lumDistance(
                    samples[index].feature,
                    emptyReferences[parity]);

            if (rank >= EMPTY_RANK_FIRST
                    && rank <= EMPTY_RANK_LAST) {
                emptyDistances[emptyCount++] = distance;
            } else {
                occupiedDistances[occupiedCount++] = distance;
            }
        }

        float emptyP90 = percentile(
                emptyDistances, emptyCount, 0.90f);
        float occupiedP10 = percentile(
                occupiedDistances, occupiedCount, 0.10f);

        float separation = occupiedP10 - emptyP90;
        float requiredSeparation =
                Math.max(0.01f, emptyP90 * 0.10f);

        if (separation <= requiredSeparation) {
            Log.i(TAG,
                    "bootstrap reject: insufficient feature separation"
                    + " emptyP90=" + emptyP90
                    + " occupiedP10=" + occupiedP10);
            return false;
        }

        templates = samples;
        labels = startingLabels;
        bootstrapSide = side;
        emptyFeaturesByParity = emptyReferences;

        emptyDistanceThreshold =
                emptyP90 + separation * 0.5f;

        float whiteSum = 0f;
        float blackSum = 0f;
        int whiteCount = 0;
        int blackCount = 0;

        for (int index = 0; index < 64; index++) {
            char piece = labels[index];
            if (piece == '.') continue;

            if (Character.isUpperCase(piece)) {
                whiteSum += samples[index].meanLum;
                whiteCount++;
            } else {
                blackSum += samples[index].meanLum;
                blackCount++;
            }
        }

        whitePieceLumAvg = whiteCount > 0
                ? whiteSum / whiteCount : 0.7f;
        blackPieceLumAvg = blackCount > 0
                ? blackSum / blackCount : 0.3f;

        previousFrame = samples;
        previousLabels = Arrays.copyOf(labels, labels.length);

        float maxEmptyNoise = 0f;

        for (int i = 0; i < 64; i++) {
            if (labels[i] != '.') continue;

            for (int j = i + 1; j < 64; j++) {
                if (labels[j] != '.'
                        || squareParity(i) != squareParity(j)) {
                    continue;
                }

                float distance = SquareSampler.lumDistance(
                        samples[i].feature,
                        samples[j].feature);

                maxEmptyNoise = Math.max(
                        maxEmptyNoise, distance);
            }
        }

        frameChangeThreshold =
                Math.max(0.3f, maxEmptyNoise * 2f + 0.1f);

        trained = true;

        Log.i(TAG, "BOOTSTRAPPED side=" + side
                + " emptyP90=" + emptyP90
                + " occupiedP10=" + occupiedP10
                + " emptyThreshold=" + emptyDistanceThreshold
                + " frameChangeThr=" + frameChangeThreshold);

        return true;
    }

    private float[][] buildEmptyReferences(
            SquareSampler.Sample[] samples) {

        int featureCount = SquareSampler.N * SquareSampler.N;
        float[][] references = new float[2][featureCount];
        int[] counts = new int[2];

        for (int rank = EMPTY_RANK_FIRST;
                rank <= EMPTY_RANK_LAST; rank++) {

            for (int file = 0; file < 8; file++) {
                int index = rank * 8 + file;
                int parity = squareParity(index);
                float[] feature = samples[index].feature;

                for (int i = 0; i < featureCount; i++) {
                    references[parity][i] += feature[i];
                }

                counts[parity]++;
            }
        }

        for (int parity = 0; parity < 2; parity++) {
            if (counts[parity] == 0) return null;

            for (int i = 0; i < featureCount; i++) {
                references[parity][i] /= counts[parity];
            }
        }

        return references;
    }

    private float percentile(
            float[] values, int count, float fraction) {

        if (count <= 0) return 0f;

        float[] sorted = Arrays.copyOf(values, count);
        Arrays.sort(sorted);

        int index = Math.min(count - 1,
                Math.max(0,
                        (int) Math.ceil(fraction * count) - 1));

        return sorted[index];
    }

    private int squareParity(int index) {
        return (((index / 8) + (index % 8)) & 1);
    }

    public synchronized char[][] classifyBoard(
            Bitmap bmp, BoardDetector.Board board) {

        if (!trained || bmp == null || board == null) {
            return null;
        }

        SquareSampler.Sample[] samples = sampleAll(bmp, board);
        if (samples == null) return null;

        char[][] grid = new char[8][8];

        for (int rank = 0; rank < 8; rank++) {
            for (int file = 0; file < 8; file++) {
                int index = rank * 8 + file;

                if (previousFrame != null
                        && previousLabels != null) {

                    float frameDistance =
                            SquareSampler.lumDistance(
                                    samples[index].feature,
                                    previousFrame[index].feature);

                    if (frameDistance < frameChangeThreshold) {
                        grid[rank][file] = previousLabels[index];
                        continue;
                    }
                }

                grid[rank][file] =
                        classifyOne(samples[index], rank, file);
            }
        }

        return grid;
    }

    public synchronized void commitConfirmed(
            char[][] grid, Bitmap bmp,
            BoardDetector.Board board) {

        if (!trained || grid == null
                || bmp == null || board == null) {
            return;
        }

        SquareSampler.Sample[] samples = sampleAll(bmp, board);
        if (samples == null) return;

        previousFrame = samples;

        if (previousLabels == null) {
            previousLabels = new char[64];
        }

        for (int rank = 0; rank < 8; rank++) {
            for (int file = 0; file < 8; file++) {
                previousLabels[rank * 8 + file] =
                        grid[rank][file] == 0
                                ? '.' : grid[rank][file];
            }
        }
    }

    private SquareSampler.Sample[] sampleAll(
            Bitmap bmp, BoardDetector.Board board) {

        if (bmp == null || board.square < 8) return null;

        int radius = Math.max(
                4, (int) (board.square * 0.32f));

        SquareSampler.Sample[] samples =
                new SquareSampler.Sample[64];

        for (int rank = 0; rank < 8; rank++) {
            for (int file = 0; file < 8; file++) {

                int centerX = Math.round(
                        board.left + (file + 0.5f) * board.square);

                int centerY = Math.round(
                        board.top + (rank + 0.5f) * board.square);

                samples[rank * 8 + file] =
                        SquareSampler.sample(
                                bmp, centerX, centerY, radius);
            }
        }

        return samples;
    }

    private char[] flipLabels(char[] source) {
        char[] flipped = new char[64];

        for (int rank = 0; rank < 8; rank++) {
            for (int file = 0; file < 8; file++) {
                flipped[rank * 8 + file] =
                        source[(7 - rank) * 8 + (7 - file)];
            }
        }

        return flipped;
    }

    private char classifyOne(
            SquareSampler.Sample sample, int rank, int file) {

        int parity = (rank + file) & 1;

        float emptyDistance = SquareSampler.lumDistance(
                sample.feature,
                emptyFeaturesByParity[parity]);

        if (emptyDistance <= emptyDistanceThreshold) {
            return '.';
        }

        float whiteDistance =
                Math.abs(sample.meanLum - whitePieceLumAvg);

        float blackDistance =
                Math.abs(sample.meanLum - blackPieceLumAvg);

        boolean isWhitePiece = whiteDistance < blackDistance;

        float bestDistance = Float.MAX_VALUE;
        char bestPiece = isWhitePiece ? 'P' : 'p';

        for (int index = 0; index < templates.length; index++) {
            char templateLabel = labels[index];

            if (templateLabel == '.') continue;

            if (Character.isUpperCase(templateLabel)
                    != isWhitePiece) {
                continue;
            }

            if (squareParity(index) != parity) continue;

            float distance = SquareSampler.lumDistance(
                    sample.feature,
                    templates[index].feature);

            if (distance < bestDistance) {
                bestDistance = distance;
                bestPiece = templateLabel;
            }
        }

        return bestPiece;
    }

    public synchronized void reset() {
        templates = null;
        labels = null;
        previousFrame = null;
        previousLabels = null;
        emptyFeaturesByParity = null;

        emptyDistanceThreshold = 0f;
        frameChangeThreshold = 0f;
        whitePieceLumAvg = 0f;
        blackPieceLumAvg = 0f;

        trained = false;
        bootstrapSide = null;
    }
            }
