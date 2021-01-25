#Youtube

#imports
import matplotlib.pyplot as plt

#Global Variables
x = []

for i in range(1,12):
	x.append(i)

#Slope function
def slope(a):
	net_slope = 0
	i = 0
	p_s = 0
	while i < 10:
		y1,y2 = a[i],a[i+1]
		x1,x2 = x[i],x[i+1]
		s = (y2-y1)/(x2-x1)
		if s < p_s:
			net_slope = net_slope - abs(s)
		else:
			net_slope = net_slope + abs(s)
		p_s = s
		i += 1
	return(net_slope)

#Function to draw graph
def draw_graph(t,vd,ld,dd):
	#Views in blue
	plt.plot(x,vd,color="blue")
	#Likes in black
	plt.plot(x,ld,color="black")
	#Dislikes in red
	plt.plot(x,dd,color="red")
	plt.savefig('graphs\\'+t+'_graph.png')

#Function to analyze and draw graphs
def youtube_analyzer(req):
	index = 3
	views = []
	likes = []
	dislikes = []
	#Getting inputs ready
	while index < len(req):
		views.append(int(req[index]))
		likes.append(int(req[index+1]))
		dislikes.append(int(req[index+2]))
		index += 3
	#Calculating differences
	view_diff = []
	like_diff = []
	dislike_diff = []
	for i in range(0,11):
		view_diff.append(views[i+1] - views[i])
		like_diff.append(likes[i+1] - likes[i])
		dislike_diff.append(dislikes[i+1] - dislikes[i])	
	#Getting net_slope for views,likes,dislikes
	net_slope_views = slope(view_diff)
	net_slope_likes = slope(like_diff)
	net_slope_dislikes = slope(dislike_diff)
	#Drawing graphs and saving it to Graphs folder with the t_id
	draw_graph(req[0],view_diff,like_diff,dislike_diff)
	
	#Return values
	return(net_slope_views,net_slope_likes,net_slope_dislikes)

def yt_verdict(v,l,d):
	final = ""
	if v > 0:
		final += "Positive hype as good surge of views seen, "
	elif v == 0:
		final += "Neutral hype, as views seem stagnant, "
	elif v < 0:
		final += "Negative hype as views haven't surged as expected', "
	
	if l > 0:
		final += "Positive hype as likes have increased, "
	elif l == 0:
		final += "Neutral hype as likes have remained stagnant, "
	elif l < 0:
		final += "Negative hype as likes haven't risen as expected', "
		
	if d > 0:
		final += "More of dislikes to the video seen."
	elif d == 0:
		final += "Dislikes have remained stagnant."
	elif d < 0:
		final += "Dislikes are low."
	
	return(final)
	