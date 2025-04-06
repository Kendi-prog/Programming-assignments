from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

#Defining a simple concave polygon(having an arrow shape)
polygon = [
    (100, 300), #point 0
    (200, 400), #point 1
    (300, 300), #point 2
    (250, 250), #point 3 (reflex vertex)
    (200, 270), #point 4
    (150, 250)  #point 5``
]


def draw_polygon(points, color):
    glColor3f(*color)
    glBegin(GL_POLYGON)
    for (x, y) in points:
        glVertex2f(x, y)
    glEnd()

def draw_split_line(p1, p2):
    glColor3f(1, 0, 0) #red color for split line
    glLineWidth(2)
    glBegin(GL_LINES)
    glVertex2f(*p1)
    glVertex2f(*p2)
    glEnd() 

def display():
    glClear(GL_COLOR_BUFFER_BIT)

    #Drawing the original polygon in gray
    draw_polygon(polygon, (0.8, 0.8, 0.8))  

    #Simulating the rotational method: split from reflex vertex (point 3) to point 0
    split_start = polygon[3]
    split_end = polygon[0]

    #Drawing the red splitting line
    draw_split_line(split_start, split_end) 

    #Defining the two resulting polygon after the split
    poly1 = [polygon[0], polygon[1], polygon[2], polygon[3]]
    poly2 = [polygon[3], polygon[4], polygon[5], polygon[0]]

    #Drawing the sub-polygons
    draw_polygon(poly1, (0.2, 0.8, 0.2)) # greenish
    draw_polygon(poly2, (0.2, 0.2, 0.8)) #blueish

    glFlush()

def init():
    glClearColor(1.0, 1.0, 1.0, 1.0) #white background
    glMatrixMode(GL_PROJECTION)
    gluOrtho2D(0, 500, 0, 500) 


def main():
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
    glutInitWindowSize(600, 600)
    glutInitWindowPosition(100, 100)
    glutCreateWindow(b"Concave Polygon Split - Rotational Method(Python)")
    init()
    glutDisplayFunc(display)
    glutMainLoop()

if __name__ == "__main__":
    main()
 