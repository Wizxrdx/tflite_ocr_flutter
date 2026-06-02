import 'package:flutter/material.dart';

/// A single point in 2D space with x and y coordinates.
class Point {
  final double x;
  final double y;

  const Point(this.x, this.y);

  /// Convert from raw coordinate list [x, y]
  factory Point.fromList(List<double> coords) {
    assert(coords.length >= 2, 'Coordinates must have at least x and y');
    return Point(coords[0], coords[1]);
  }

  /// Convert to Offset for Flutter drawing
  Offset toOffset() => Offset(x, y);

  @override
  String toString() => 'Point($x, $y)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Point &&
          runtimeType == other.runtimeType &&
          x == other.x &&
          y == other.y;

  @override
  int get hashCode => x.hashCode ^ y.hashCode;
}

/// A polygon defined by a list of points (vertices).
/// Typically represents a detected text region with arbitrary shape.
class Polygon {
  final List<Point> points;

  const Polygon(this.points);

  /// Create from raw list of [x, y] coordinate pairs
  factory Polygon.fromList(List<List<double>> coords) {
    final points = coords.map(Point.fromList).toList();
    return Polygon(points);
  }

  /// Number of vertices in the polygon
  int get pointCount => points.length;

  /// Check if polygon has enough points to be valid
  bool get isValid => pointCount >= 3;

  @override
  String toString() => 'Polygon(points: $pointCount)';
}

class Box {
  final double x;
  final double y;
  final double width;
  final double height;

  const Box(this.x, this.y, this.width, this.height);

  /// Convert from raw list [x, y, width, height]
  factory Box.fromList(List<List<double>> coords) {
    assert(coords.length >= 4, 'Box coordinates must have x, y, width, height');
    return Box(coords[0][0], coords[0][1], coords[0][2], coords[0][3]);
  }

  /// Convert to Rect for Flutter drawing
  Rect toRect() => Rect.fromLTWH(x, y, width, height);

  @override
  String toString() =>
      'Box(x: $x, y: $y, width: $width, height: $height)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Box &&
          runtimeType == other.runtimeType &&
          x == other.x &&
          y == other.y &&
          width == other.width &&
          height == other.height;

  @override
  int get hashCode =>
      x.hashCode ^ y.hashCode ^ width.hashCode ^ height.hashCode;
}

/// Result from text detection service.
/// Contains detected text regions as polygons.
class TextDetectionResult {
  final List<Box> boxes;

  const TextDetectionResult(this.boxes);

  factory TextDetectionResult.fromRawOutput(
    List<Box> rawBoxes,
  ) {
    if (rawBoxes.isEmpty) {
      return const TextDetectionResult([]);
    }

    // Median height (robust for receipts)
    final heights = rawBoxes.map((b) => b.height).toList()..sort();
    final medianHeight = heights[heights.length ~/ 2];

    // Bucket into rows
    final rows = <int, List<Box>>{};

    for (final box in rawBoxes) {
      final centerY = box.y + box.height * 0.5;
      final row = (centerY / medianHeight).round();
      (rows[row] ??= <Box>[]).add(box);
    }

    final mergedBoxes = <Box>[];

    for (final rowBoxes in rows.values) {
      if (rowBoxes.isEmpty) continue;

      rowBoxes.sort((a, b) => a.x.compareTo(b.x));

      // Start first group
      double left = rowBoxes.first.x;
      double top = rowBoxes.first.y;
      double right = rowBoxes.first.x + rowBoxes.first.width;
      double bottom = rowBoxes.first.y + rowBoxes.first.height;

      for (var i = 1; i < rowBoxes.length; i++) {
        final next = rowBoxes[i];

        final nextLeft = next.x;
        final nextRight = next.x + next.width;
        final nextTop = next.y;
        final nextBottom = next.y + next.height;

        final gap = nextLeft - right;

        final maxGap = (bottom - top) * 0.8 * 3; // your aggressive merge factor

        if (gap <= maxGap) {
          // merge horizontally + expand vertically fully
          left = left < nextLeft ? left : nextLeft;
          right = right > nextRight ? right : nextRight;

          top = top < nextTop ? top : nextTop;
          bottom = bottom > nextBottom ? bottom : nextBottom;
        } else {
          mergedBoxes.add(Box(left, top, right - left, bottom - top));

          left = nextLeft;
          top = nextTop;
          right = nextRight;
          bottom = nextBottom;
        }
      }

      // flush last group
      mergedBoxes.add(Box(left, top, right - left, bottom - top));
    }

    return TextDetectionResult(mergedBoxes);
  }

  /// Number of detected regions
  int get detectionCount => boxes.length;

  /// Check if any detections were found
  bool get hasDetections => boxes.isNotEmpty;

  @override
  String toString() => 'TextDetectionResult(detections: $detectionCount)';
}
