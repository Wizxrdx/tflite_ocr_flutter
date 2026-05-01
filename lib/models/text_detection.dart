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

/// Result from text detection service.
/// Contains detected text regions as polygons.
class TextDetectionResult {
  final List<Polygon> polygons;

  const TextDetectionResult(this.polygons);

  /// Create from raw detection output (List<List<List<double>>>)
  factory TextDetectionResult.fromRawOutput(
    List<List<List<double>>> rawPolygons,
  ) {
    final polygons = rawPolygons.map(Polygon.fromList).toList();
    return TextDetectionResult(polygons);
  }

  /// Number of detected regions
  int get detectionCount => polygons.length;

  /// Check if any detections were found
  bool get hasDetections => polygons.isNotEmpty;

  @override
  String toString() => 'TextDetectionResult(detections: $detectionCount)';
}
