import pyqtgraph as pg
import time
import math

# Create a PyQtGraph window
app = pg.mkQApp()
win = pg.PlotWidget()
win.setWindowTitle('Real-time Plot')  # Set the window title
win.setLabel('left', 'Y-Axis')  # Set the y-axis label
win.setLabel('bottom', 'X-Axis')  # Set the x-axis label
win.showGrid(x=True, y=True)  # Add a grid
win.show()

# Create multiple plot curves
curve1 = win.plot(pen='r', name='Curve 1', symbol='+')
curve2 = win.plot(pen='b', name='Curve 2', symbol='+')

# Add a legend (optional)
legend = pg.LegendItem()
win.addItem(legend)
legend.addItem(curve1, 'Curve 1')
legend.addItem(curve2, 'Curve 2')
#legend.setPos(10, 10)  # Adjust legend position

# Set manual x-axis range
win.setYRange(-2, 2)

# Simulate real-time data
x_data, y1_data, y2_data = [], [], []

# Update the plot in a loop
for i in range(1000):
    x_data.append(i)
    
    y1_data.append(math.cos(i*2*math.pi/10))
    y2_data.append(math.sin(i*2*math.pi/10))

    curve1.setData(x=x_data, y=y1_data)
    curve2.setData(x=x_data, y=y2_data)
    
    app.processEvents()  # Process events to update the plot
    time.sleep(0.02)