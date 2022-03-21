import random
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
 
class Point:
  """
  2D point class.

  Instance attributes
  -------------------

  x,y: float 
    2D coordinates

  """
  def __init__(self, x=0.0, y=0.0):
    self.x = x
    self.y = y
 
  def draw(self, color='black'):
    plt.plot(self.x, self.y, 'o', color=color)
 
  def asVector3(self):
    """Express the point as a 3D vector"""
    return np.array([self.x, 0, self.y])
 
  def move(self, deltaX=0.0, deltaY=0.0):
    self.x += deltaX
    self.y += deltaY
 
  def distance2d(p0, p1):
    """Returns distance between two points on a plane"""
    return (np.sqrt((p1.x - p0.x)**2 + (p1.y - p0.y)**2))
 
  def __str__(self):
    return f"Point({self.x}, {self.y})"

class Line:
  """
  2D line class.

  Instance attributes
  -------------------

  p0, p1: Points
    2D points

  """
  def __init__(self, p0, p1):
    self.p0 = p0
    self.p1 = p1
 
  def draw(self, color='red'):
    plt.plot([self.p0.x, self.p1.x],[self.p0.y, self.p1.y], color=color)
 
  def asVector3(self):
    """Express the line as a 3D vector"""
    return (self.p1.asVector3() - self.p0.asVector3())
 
  def __str__(self):
    return f"Line from: {self.p0} to {self.p1}"

class Triangle:
  """
  2D triangle class.

  Instance attributes
  -------------------

  p0, p1, p2: Points
    2D points
  pointCloud: pointCloud
    a pointCloud object that the triangle is a part of
  Tid: int
    triangle ID within the pointCloud
  pc: Point
    geometric center of the triangle
  neighbors: [Tid, Tid, Tid]
    list of neighboring triangle indeces

  """
  def __init__(self, p0, p1, p2, pointCloud, Tid = -1):
    self.p0 = p0
    self.p1 = p1
    self.p2 = p2
    self.pointCloud = pointCloud
    self.pc = Point((self.pointCloud[self.p0].x + self.pointCloud[self.p1].x + self.pointCloud[self.p2].x) / 3, 
                    (self.pointCloud[self.p0].y + self.pointCloud[self.p1].y + self.pointCloud[self.p2].y) / 3)
    self.neighbors = [None, None, None] # [p0p1, p1p2, p2p0]
    self.Tid = Tid
    self.sort()
 
  def sort(self):
    """Sorts triangle points counterclockwise (for Delaunay triangulation)"""
    vector_pcp0 = Line(self.pc, self.pointCloud[self.p0]).asVector3()
    vector_pcp1 = Line(self.pc, self.pointCloud[self.p1]).asVector3()
 
    if np.cross(vector_pcp0, vector_pcp1)[1] > 0: 
      self.p0, self.p1 = self.p1, self.p0 # it's enough to flip two adjacent points once
 
  def getEdges(self):
    return ((self.p0, self.p1), (self.p1, self.p2), (self.p2, self.p0))
 
  def addNeighbor(self, T):
    """Mutually adds two triangles to their neighbor lists"""
    thisEdges = self.getEdges()
    otherEdges = T.getEdges()
    for thisEdge_index in range(len(thisEdges)):
      for otherEdge_index in range(len(otherEdges)):
        if (thisEdges[thisEdge_index][0], thisEdges[thisEdge_index][1]) == \
        (otherEdges[otherEdge_index][1], otherEdges[otherEdge_index][0]):
          self.neighbors[thisEdge_index] = T
          #print(f'T{self.Tid} has a neighbor T{T.Tid} at {thisEdge_index}')
          T.neighbors[otherEdge_index] = self
          #print(f'T{T.Tid} has a neighbor T{self.Tid} at {otherEdge_index}')
          return
 
    return
 
  def checkNeighbor(self):
    """Checks the neighbor list for its consistency"""
    for this_neighbor_index in range(len(self.neighbors)):
      counter = 0
      if self.neighbors[this_neighbor_index]:
        for other_neighbor_index in range(len(self.neighbors[this_neighbor_index].neighbors)):
          if self.neighbors[this_neighbor_index].neighbors[other_neighbor_index] == self:
            counter += 1
        assert counter == 1, f'Neighbor listing error! (T{self.Tid}, T{self.neighbors[this_neighbor_index].Tid})'
 
    return
        
  def circumcircleContains(self, point_index):
    """Check if a point is within a triangle's circumcircle"""
    a = self.pointCloud[self.p0]
    b = self.pointCloud[self.p1]
    c = self.pointCloud[self.p2]
    d = self.pointCloud[point_index]
    det = np.linalg.det([[a.x, a.y, a.x**2 + a.y**2, 1],
                         [b.x, b.y, b.x**2 + b.y**2, 1],
                         [c.x, c.y, c.x**2 + c.y**2, 1],
                         [d.x, d.y, d.x**2 + d.y**2, 1]])
    
    return det > 1E-8
 
  def circumcircle(self):
    """Find the triangle's circumcircle"""
    a = self.pointCloud[self.p0]
    b = self.pointCloud[self.p1]
    c = self.pointCloud[self.p2]
    a0 = Point(0., 0.)
    b0 = Point(b.x - a.x, b.y - a.y)
    c0 = Point(c.x - a.x, c.y - a.y)
    
    D = 2 * (b0.x * c0.y - b0.y * c0.x)
    u0x = (c0.y * (b0.x**2 + b0.y**2) - b0.y * (c0.x**2 + c0.y**2)) / D
    u0y = (b0.x * (c0.x**2 + c0.y**2) - c0.x * (b0.x**2 + b0.y**2)) / D
 
    radius = np.sqrt(u0x**2 + u0y**2)
    circumcenter = Point(u0x + a.x, u0y + a.y)
 
    return circumcenter, radius
 
  def flipWithNeighbor(self, neighbor_index):
    """Perform an edge flip with a neighboring triangle"""
    neighbor = self.neighbors[neighbor_index]
    #print(f'Flipping T{self.Tid} and T{neighbor.Tid}...')
    vertices = set([self.p0, self.p1, self.p2]) | set([neighbor.p0, neighbor.p1, neighbor.p2])
    commonEdge_set = set(self.getEdges()[neighbor_index])
    commonEdge = [*commonEdge_set]
    externalPoints_set = vertices - commonEdge_set
    externalPoints = [*externalPoints_set]
    assert len(externalPoints) == 2, 'A pair of adjacent triangles must have exactly TWO external points!'
    T1_new = Triangle(externalPoints[0], externalPoints[1], commonEdge[0], self.pointCloud, Tid = self.Tid)
    T2_new = Triangle(externalPoints[0], externalPoints[1], commonEdge[1], self.pointCloud, Tid = neighbor.Tid)
    old_triangles = set([self, neighbor])
    triangle_neighborhood = (set(self.neighbors) | set(neighbor.neighbors)) - old_triangles - set([None]) 
    #print(f'T{self.Tid} and T{neighbor.Tid} triangle neighborhood: {[str(T) for T in triangle_neighborhood]}')
    for triangle in triangle_neighborhood:
      T1_new.addNeighbor(triangle)
      T2_new.addNeighbor(triangle)
 
    for triangle in triangle_neighborhood:
      for neighbor in triangle.neighbors:
        assert neighbor not in old_triangles, 'Old triangles have to be excluded from the updated neighbor lists!'
 
    T1_new.addNeighbor(T2_new)
 
    return T1_new, T2_new
 
  def area(self):
    """Find the area of the triangle"""
    a = self.pointCloud[self.p0]
    b = self.pointCloud[self.p1]
    c = self.pointCloud[self.p2]
    return 0.5*abs(a.x * (b.y - c.y) + 
                   b.x * (c.y - a.y) + 
                   c.x * (a.y - b.y))
 
  def isWithinTriangle(self, point_index):
    """Check if a point is within a triangle"""
    T1 = Triangle(point_index, self.p0, self.p1, self.pointCloud)
    T2 = Triangle(point_index, self.p1, self.p2, self.pointCloud)
    T3 = Triangle(point_index, self.p0, self.p2, self.pointCloud)
 
    A = self.area()
    A1 = T1.area()
    A2 = T2.area()
    A3 = T3.area()
 
    if abs((A1 + A2 + A3) - A) < 1E-9:
      return True
    else:
      return False
  
  def draw(self, color='blue', showTid = False):
    if showTid:
      plt.text(x = self.pc.x, y = self.pc.y, s = self.Tid)
    Line(self.pointCloud[self.p0], self.pointCloud[self.p1]).draw(color)
    Line(self.pointCloud[self.p1], self.pointCloud[self.p2]).draw(color)
    Line(self.pointCloud[self.p2], self.pointCloud[self.p0]).draw(color)
 
  def drawCircumcircle(self, color='green', drawCenter=False):
    circumcenter, radius = self.circumcircle()
    if drawCenter:
      circumcenter.draw(color=color)
    phi = np.linspace(0, 2 * np.pi, 100)
    x = circumcenter.x + radius * np.cos(phi)
    y = circumcenter.y + radius * np.sin(phi)
    plt.plot(x, y, color=color)
 
  def __str__(self):
    return f"T{self.Tid}({self.p0}, {self.p1}, {self.p2}); \
              Neighbors: {self.neighbors[0].Tid if self.neighbors[0] else '-'},\
              {self.neighbors[1].Tid if self.neighbors[1] else '-'},\
              {self.neighbors[2].Tid if self.neighbors[2] else '-'}"

class PointCloud:
  """
  Collection of 2D points class.

  Instance attributes
  -------------------

  N: int
    number of points in the pointCloud
  min_distance: float
    minimal 2D distance between each pair of points
  max_attempts: int
    maximum number of attempts to add a new point
  convex_hull_indices: list of ints
    contains indices of the convex hull points
  points: list of Points
    list of all points in the pointCloud
  
  """
  def __init__(self, N=0, min_distance=0.05, max_attempts=50000):
    self.N = N
    self.convex_hull_indices = []
    self.points = []
    
    # Put the points so that there is some distance between them
    if self.N > 0:
      self.points = [Point(random.random(), random.random())]
      i = 1
      attempts = 0
      while i < self.N:
        attempts += 1
        probe = Point(random.random(), random.random())
        distances = []
        for j in range(0, i):
          distances.append(probe.distance2d(self.points[j]))
        if min(distances) > min_distance:
          self.points.append(probe)
          i += 1
        assert attempts < max_attempts, f'Cannot place {N} points\
  within {min_distance} from each other in {max_attempts} attempts!'
 
  def __getitem__(self, index):
    return self.points[index]
 
  def __len__(self):
    return len(self.points)  
    
  def drawPoints(self, color='black'):
    for point in self.points:
        point.draw(color)
 
  def addPoint(self, point):
    assert type(point) is Point, 'Can add objects of type Point only.'
    self.points.append(point)
    self.N += 1
    
  def removePoint(self, position):
    del self.points[position]
    self.N -= 1
    
  def clearPoints(self):
    self.points = []
    self.N = 0
 
  def __str__(self):
    point_list = ""
    for point in self.points:
      point_list += str(point) + "\n"
    return f"{self.N} points: \n" + point_list
 
  def nextConvexHullPoint(self, start_point_index, start_direction):
      """Returns the next convex hull point index"""
      dot_products = []
      point_indices = []
      for point_index in range(len(self.points)):
        if point_index == start_point_index:
          continue
        else:
          edge = Line(self.points[start_point_index], self.points[point_index])
          next_direction = edge.asVector3() / np.linalg.norm(edge.asVector3())
          cross_product = np.cross(start_direction, next_direction)
          dot_product = np.dot(start_direction, next_direction)
          if cross_product[1] >= 0:
            dot_products.append(dot_product)
            point_indices.append(point_index)
 
      return point_indices[dot_products.index(max(dot_products))]
 
  def getConvexHull(self):
    """Returns the indices of convex hull points"""
 
    for point in self.points:
      assert type(point) is Point, 'Input list contains objects of types other than Point.'
    
    assert len(self.points) > 2, 'Not enough points to construct the convex hull!'
 
    points_X = [point.x for point in self.points]
    minXpoint_index = points_X.index(min(points_X))
    minXpoint = self.points[minXpoint_index]
 
    start_point_index = minXpoint_index
    current_point_index = start_point_index
    end_point_index = -1
 
    start_direction = np.array([0, 0, 1])
    current_direction = start_direction
 
    self.convex_hull_indices = [start_point_index]
 
    num_iterations = 0
 
    while start_point_index != end_point_index and num_iterations < len(self.points):
      end_point_index = self.nextConvexHullPoint(current_point_index, current_direction)
      num_iterations += 1
      
      self.convex_hull_indices.append(end_point_index)
      
      current_direction = Line(self.points[current_point_index], self.points[end_point_index]).asVector3()
      current_point_index = end_point_index
 
    del self.convex_hull_indices[-1] # Remove the starting point
 
    return self.convex_hull_indices
 
  def drawConvexHull(self, pngname = 'convex.png', point_color = 'black', convex_hull_color = 'red'):
    self.drawPoints(point_color)
 
    if len(self.convex_hull_indices) == 0:
      self.getConvexHull()
    
    for i in range(len(self.convex_hull_indices)):
      if i < len(self.convex_hull_indices) - 1:
        edge = Line(self.points[self.convex_hull_indices[i]], self.points[self.convex_hull_indices[i + 1]])
        edge.draw(convex_hull_color)
      else:
        edge = Line(self.points[self.convex_hull_indices[i]], self.points[self.convex_hull_indices[0]])
        edge.draw(convex_hull_color)

    plt.savefig(pngname, dpi=300)
    plt.show()
 
    return

class Triangulation:
  """
  Delaunay triangulation class.

  Instance attributes
  -------------------

  pointCloud: pointCloud
    a pointCloud object for triangulation
  triangles: list of triangles
    a list of triangles built from the pointCloud
  
  """
  def __init__(self, pointCloud):
    self.pointCloud = pointCloud
    self.triangles = []
    self.initialTriangulation()
    self.createNeighborList()
 
  def __getitem__(self, index):
    return self.triangles[index]
 
  def __setitem__(self, index, T):
    self.triangles[index] = T
    return
 
  def __len__(self):
    return len(self.triangles)
 
  def earClipping(self):
    """Triangulation of point cloud convex hull using the ear clipping algorithm"""
    if len(self.pointCloud.convex_hull_indices) == 0:
      self.pointCloud.getConvexHull()
    convex_hull_indices_current = self.pointCloud.convex_hull_indices.copy()
 
    while len(convex_hull_indices_current) > 3:
      ear = Triangle(convex_hull_indices_current[0], convex_hull_indices_current[1], \
                    convex_hull_indices_current[2], self.pointCloud)
      self.triangles.append(ear)
 
      del convex_hull_indices_current[1]
 
    if len(convex_hull_indices_current) == 3:
      last_ear = Triangle(convex_hull_indices_current[0], convex_hull_indices_current[1], \
                        convex_hull_indices_current[2], self.pointCloud)
      self.triangles.append(last_ear)
 
    return 
 
  def initialTriangulation(self):
    """Construct initial triangulation of the point cloud using the ear clipping method
    and connecting a point inside a triangle with the triangle vertices until there are no free points left"""
    self.earClipping()
    free_points = [point_index for point_index in range(len(self.pointCloud)) \
                  if point_index not in self.pointCloud.convex_hull_indices]
 
    while len(free_points) > 0:
      for point_index in free_points:
        areas = []
        for triangle_index in range(len(self)):
          if self[triangle_index].isWithinTriangle(point_index):
            areas.append((self[triangle_index].area(), triangle_index))
        assert len(areas) == 1, 'Triangles cannot intersect!'
 
        if len(areas) > 0:
          selected_triangle = self[min(areas)[1]]
          T1 = Triangle(point_index, selected_triangle.p0, selected_triangle.p1, self.pointCloud)
          self.triangles.append(T1)
          T2 = Triangle(point_index, selected_triangle.p1, selected_triangle.p2, self.pointCloud)
          self.triangles.append(T2)
          T3 = Triangle(point_index, selected_triangle.p0, selected_triangle.p2, self.pointCloud)
          self.triangles.append(T3)
          free_points.remove(point_index)
          self.triangles.remove(selected_triangle)
 
    for i in range(len(self)):
      self[i].Tid = i
 
    return
 
  def createNeighborList(self):
    for i in range(len(self)):
      for j in range(i):
        self[i].addNeighbor(self[j])
 
    for i in range(len(self)):
      self[i].checkNeighbor()
 
    return
 
  def makeDelaunay(self, showSteps = False):
    """Flip triangles until they become Delaunay"""
    while not self.delaunayTriangulationCheck():
      for triangle_index in range(len(self)):
        this_points = set([self[triangle_index].p0, self[triangle_index].p1, self[triangle_index].p2])
        for this_neighbor_index in range(len(self[triangle_index].neighbors)):
          neighbor = self[triangle_index].neighbors[this_neighbor_index]
          if self[triangle_index].neighbors[this_neighbor_index]:
            neighbor_points = set([neighbor.p0, neighbor.p1, neighbor.p2])
            external_point_set = neighbor_points - this_points
            external_point = [*external_point_set][0]
            assert len(external_point_set) == 1, 'A neighbor must have exactly ONE external point!'
            if self[triangle_index].circumcircleContains(external_point):
              old_neighbor_index = self[triangle_index].neighbors[this_neighbor_index].Tid
              self[triangle_index], self[old_neighbor_index] = self[triangle_index].flipWithNeighbor(this_neighbor_index)
 
              if showSteps:
                for t in self:
                  t.draw(drawText=True)
                  print(t)
                self.pointCloud.drawConvexHull()
              break
    return
 
  def delaunayTriangulationCheck(self):
    """Checks if the triangles are Delaunay"""
    for point_index in range(len(self.pointCloud)):
      for triangle in self.triangles:
        if triangle.circumcircleContains(point_index):
          #print(f'Point({point_index}) is within T{triangle.Tid}')
          return False
 
    return True

  def drawTriangulation(self, pngname='delaunay.png', point_color='black', convex_hull_color='red', 
                        triangle_color='blue', showTid=False, showCircumcircles = False, circ_color='green'):
    if not showCircumcircles:
      for triangle in self.triangles:
        triangle.draw(color=triangle_color, showTid=showTid)
      self.pointCloud.drawConvexHull(pngname, point_color, convex_hull_color)
    else:
      for triangle in self.triangles:
        triangle.draw(color=triangle_color, showTid=showTid)
        triangle.drawCircumcircle(color=circ_color)
      self.pointCloud.drawConvexHull(pngname, point_color, convex_hull_color)
    return

class Voronoi:
  """
  Unfinished class for building Voronoi diagrams
  from Delaunay triangulation
  """
  def __init__(self, triangulation):
    self.triangles = triangulation
    self.vertices = []
    for triangle in self.triangles:
      self.vertices.append(triangle.circumcircle()[0])
    self.edges = []
    for vertex_index in range(len(self.vertices)):
      for neighbor in self.triangles[vertex_index]:
        if neighbor:
          self.edges.append((vertex_index, neighbor.Tid))
        else:
          self.edges.append((vertex_index, neighbor.Tid))
    return
