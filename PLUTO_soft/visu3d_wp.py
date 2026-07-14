from subprocess import call
import sys
sys.path.append("/Applications/VisIt.app/Contents/Resources/2.10.1/darwin-x86_64/lib/site-packages")
import os
from visit import *

def myVisitExpressions():
  vE = []
  vE.append(("R",     "coord(mesh)[0]"))
  vE.append(("Theta", "coord(mesh)[1]"))
  vE.append(("Phi",   "coord(mesh)[2]"))
  vE.append(("Vr",    "vx1"))
  vE.append(("Vth",   "vx2"))
  vE.append(("Vphi",  "vx3"))
  vE.append(("Br",    "bx1 + B1_bckg"))
  vE.append(("Bth",   "bx2 + B2_bckg"))
  vE.append(("Bphi",  "bx3 + B3_bckg"))
  vE.append(("vv",    "sqrt(Vr^2 + Vth^2 + Vphi^2)"))
  vE.append(("bb",    "sqrt(Br^2 + Bth^2 + Bphi^2)"))
  vE.append(("va",    "bb/(sqrt(rho) + 1.0e-10)"))
  vE.append(("Alf",   "vv/(va + 1.0e-10)"))
  vE.append(("VX",    "sin(Theta)*cos(Phi)*Vr + cos(Theta)*cos(Phi)*Vth - sin(Phi)*Vphi"))
  vE.append(("VY",    "sin(Theta)*sin(Phi)*Vr + cos(Theta)*sin(Phi)*Vth + cos(Phi)*vphi"))
  vE.append(("VZ",    "cos(Theta)*Vr - sin(Theta)*Vth"))
  vE.append(("BX",    "sin(Theta)*cos(Phi)*Br + cos(Theta)*cos(Phi)*Bth - sin(Phi)*Bphi"))
  vE.append(("BY",    "sin(Theta)*sin(Phi)*Br + cos(Theta)*sin(Phi)*Bth + cos(Phi)*Bphi"))
  vE.append(("BZ",    "cos(Theta)*Br - sin(Theta)*Bth"))
  vE.append(("cs2",   "1.05*prs/rho"))
  vE.append(("Apol2",  "(Br^2 + Bth^2)/rho"))
  vE.append(("Aphi2",  "Bphi^2/rho"))
  vE.append(("Alfs",   "2.0*(Vr^2+Vth^2)/(cs2+Apol2+Aphi2 - sqrt((cs2+Apol2+Aphi2)^2-4.0*cs2*Apol2))"))
  vE.append(("Alff",   "2.0*(Vr^2+Vth^2)/(cs2+Apol2+Aphi2 + sqrt((cs2+Apol2+Aphi2)^2-4.0*cs2*Apol2))"))
  return vE

# User editable
it = "100"
wdir = "./VTK"
case = "./"
fname = wdir + "/" + case + "/" + "data.0" + it + ".vtr"
Bmax = 1.0e-1
outdir = "./figs/"
outname = 'wp_'

# Create output dir and simulation open file
from subprocess import call
call("mkdir -pv" + outdir, shell=True)
OpenDatabase(fname)

# Define expressions
for (a,b) in myVisitExpressions():
  DefineScalarExpression(a,b)
DefineVectorExpression("VelField", "{Vr, Vth, Vphi}")
DefineVectorExpression("MagField", "{Br, Bth, Bphi}")
DefineVectorExpression("VEL", "{VX, VY, VZ}")
DefineVectorExpression("MAG", "{BX, BY, BZ}")

# Ecliptic plane with velocity
AddPlot("Pseudocolor", "vv")
ps = PseudocolorAttributes()
ps.opacityType = 2 ; ps.opacity = 0.75
ps.legendFlag = 0
SetPlotOptions(ps)
AddOperator("Transform")
tf = TransformAttributes()
tf.transformType = 1 ; tf.inputCoordSys = 2 ; tf.outputCoordSys = 0
SetOperatorOptions(tf)
AddOperator("Slice")
sl = SliceAttributes()
sl.normal = (0., 0.13, 0.99); sl.project2d = 0
SetOperatorOptions(sl)

# Stellar surface [Br]
AddPlot("Pseudocolor", "Br")
ps = PseudocolorAttributes()
ps.minFlag = 1; ps.maxFlag = 1; ps.min = -1.0*Bmax; ps.max = Bmax
ps.colorTableName="RdBu";ps.invertColorTable=1
ps.legendFlag = 0
SetPlotOptions(ps)
AddOperator("Slice")
ss = SliceAttributes()
ss.originIntercept = 1.0 ; ss.axisType = 0 ; ss.project2d = 0
SetOperatorOptions(ss)
AddOperator("Transform")
tf = TransformAttributes()
tf.transformType = 1 ; tf.inputCoordSys = 2 ; tf.outputCoordSys = 0
SetOperatorOptions(tf)

# Alfven surface
AddPlot("Contour","Alf")
ct = ContourAttributes()
ct.contourMethod = ct.Value; ct.contourValue = (1.0)
ct.colorType = ct.ColorBySingleColor; ct.singleColor = (255, 222, 173, 70) #(255, 255, 128, 0)
#ct.contourPercent = (50)
ct.legendFlag = 0
SetPlotOptions(ct)
AddOperator("Transform")
tf = TransformAttributes()
tf.transformType = 1 ; tf.inputCoordSys = 2 ; tf.outputCoordSys = 0
SetOperatorOptions(tf)

# Magneto-sonic surfaces
# Slow
#AddPlot("Contour","Alfs")
#ct = ContourAttributes()
#ct.contourMethod = ct.Value; ct.contourValue = (1.0)
#ct.colorType = ct.ColorBySingleColor; ct.singleColor = (255, 153, 76, 0)
#ct.contourPercent = (50)
#ct.legendFlag = 0
#SetPlotOptions(ct)
#AddOperator("Transform")
#tf = TransformAttributes()
#tf.transformType = 1 ; tf.inputCoordSys = 2 ; tf.outputCoordSys = 0
#SetOperatorOptions(tf)
# Fast 
#AddPlot("Contour","Alff")
#ct = ContourAttributes()
#ct.contourMethod = ct.Value; ct.contourValue = (1.0)
#ct.colorType = ct.ColorBySingleColor; ct.singleColor = (255, 255, 204, 153)
#ct.contourPercent = (50)
#ct.legendFlag = 0
#SetPlotOptions(ct)
#AddOperator("Transform")
#tf = TransformAttributes()
#tf.transformType = 1 ; tf.inputCoordSys = 2 ; tf.outputCoordSys = 0
#SetOperatorOptions(tf)

# Magnetic field lines
AddPlot("Streamline","MagField")
sl = StreamlineAttributes()
sl.sourceType = sl.SpecifiedSphere; sl.radius = 10.0
sl.sampleDensity0 = 5; sl.sampleDensity1 = 5;
sl.integrationDirection = sl.Both
sl.coloringMethod = sl.ColorByVariable; sl.coloringVariable = "Br"; sl.colorTableName = "RdBu"
sl.legendMinFlag = 1; sl.legendMaxFlag = 1
sl.legendMin = -0.001; sl.legendMax = 0.001
sl.displayMethod = sl.Tubes; sl.tubeRadiusBBox = 0.0012;
sl.showSeeds = 0; sl.legendFlag = 0
sl.fillInterior = 0
SetPlotOptions(sl)
AddOperator("Transform")
tf = TransformAttributes()
tf.transformType = 1 ; tf.inputCoordSys = 2 ; tf.outputCoordSys = 0
SetOperatorOptions(tf)

# Light
light = GetLight(0)
light.enabledFlag = 1
light.type = 1
SetLight(0,light)

# Annotations
a = AnnotationAttributes()
a.axes3D.visible = 1 ; a.axes3D.bboxFlag = 1
a.userInfoFlag = 0 ; a.databaseInfoFlag = 0 ; a.backgroundColor = (255, 255, 255, 255)
SetAnnotationAttributes(a)

# Draw
DrawPlots()
v = GetView3D()
v.viewNormal = (0.008, 0.937, 0.350)
v.viewUp = (-0.01, -0.35, 0.94)
SetView3D(v)

# Save
sw = SaveWindowAttributes()
sw.format = sw.PNG
sw.fileName = outname
sw.outputToCurrentDirectory = 0 ; sw.outputDirectory = outdir
sw.width = 1920 ; sw.height = 1808 ; sw.screenCapture = 0
sw.stereo = 0
SetSaveWindowAttributes(sw)
saveW = SaveWindow()
