import h5py  
import numpy as np
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sys
import glob,pdb
import plotly,copy
import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('tejasdkulkarni', 'uog8mj0pfk')
import sys
import plotly.graph_objs as go
from numpy import linalg as LA

total_goals = 6

def get_layout(xtitle='Steps', ytitle='Average Extrinsic Reward'):

	layout = Layout(
		legend=dict(
		        x=0.74,
		        y=0.1,
				font=dict(
		            family='sans-serif',
		            size=40,
	    	        color='#000'
		        ),
				borderwidth=1
		    ),

	    paper_bgcolor='rgb(255,255,255)',
	    plot_bgcolor='rgb(229,229,229)',
	    xaxis=XAxis(
	    	title=xtitle,#(x10000 interactions)',
	    	titlefont=dict(
	    		size=40,
			),
			tickfont=dict(
				size=20
			),
	        gridcolor='rgb(255,255,255)',
	        # range=[1,10],
	        showgrid=True,
	        showline=False,
	        showticklabels=True,
	        # tickcolor='rgb(127,127,127)',
	        # ticks='outside',
	        zeroline=False
	    ),
	    yaxis=YAxis(
			# x=0.84,
			# y=1,
	    	title=ytitle,
	    	titlefont=dict(
	    		size=40,
			),
			tickfont=dict(
				size=20
			),
			# range=[0,1],
	        gridcolor='rgb(255,255,255)',
	        showgrid=True,
	        showline=False,
	        showticklabels=True,
	        # tickcolor='rgb(127,127,127)',
	        ticks='inside',
	        zeroline=False
	    ),
	)
	return layout

def get_reward_plot_realexp():
	global total_goals
	data_basic = np.loadtxt('logs/golden/test_avgR.log')
	data_db=dict();cnt=0
	for exp in glob.glob('logs/latest/*'):
		data = np.loadtxt(exp+'/test_avgR.log')
		# data = np.hstack((data_basic,data))
		data_db[cnt]=data
		# print(min(data_db[cnt]))
		cnt=cnt+1

	minlen=np.inf
	for i in range(len(data_db)):
		minlen = min(len(data_db[i]), minlen)

	y=np.zeros(minlen)
	agg=np.zeros((len(data_db),minlen))
	for i in range(len(data_db)):
		y = y + data_db[i][0:minlen]
		agg[i,:] = data_db[i][0:minlen]


	y=(1.0*y)/len(data_db)
	std = np.std(agg,axis=0)
	y_lower = y-std
	y_lower=y_lower[::-1]
	y_upper = y+std
	x= np.arange(minlen)
	x = x*30000 + 30000 #learn start offset and each epoch is after 30000 steps

	x_rev = x[::-1]

	y_base = list(y*0)
	x=list(x);x_rev=list(x_rev);y=list(y);y_lower=list(y_lower);y_upper=list(y_upper)

	trace1 = Scatter(
	    x=x+x_rev,
	    y=y_upper+y_lower,
	    fill='tozerox',
	    fillcolor='rgba(0,176,246,0.2)',
	    line=Line(color='transparent'),
	    name='Our Approach',
	    showlegend=False,
	)

	trace2 = Scatter(
	    x=x,
	    y=y,
	    line=Line(color='rgb(0,176,246)'),
	    mode='lines',
	    name='Our Approach',
	)

	trace3 = Scatter(
	    x=x,
	    y=y_base,
	    line=Line(color='rgb(255,0,0)'),
	    mode='lines',
	    name='DQN',
	)


	data = Data([trace1, trace2, trace3])
	fig = Figure(data=data, layout=get_layout())
	plotly.offline.plot(fig, filename='Reward')




def get_reward_plot_synthetic(name, path,r,g,b):
	data_db=dict();cnt=0
	for exp in glob.glob(path+'/*'):
		data = np.loadtxt(exp+'/test_avgR.log')
		data_db[cnt]=data
		cnt=cnt+1

	minlen=np.inf
	for i in range(len(data_db)):
		minlen = min(len(data_db[i]), minlen)

	y=np.zeros(minlen)
	agg=np.zeros((len(data_db),minlen))
	for i in range(len(data_db)):
		y = y + data_db[i][0:minlen]
		agg[i,:] = data_db[i][0:minlen]


	y=(1.0*y)/len(data_db)
	std = np.std(agg,axis=0)
	y_lower = y-std
	y_lower=y_lower[::-1]
	y_upper = y+std
	x= np.arange(minlen)
	x_rev = x[::-1]

	x=list(x);x_rev=list(x_rev);y=list(y);y_lower=list(y_lower);y_upper=list(y_upper)

	trace1 = Scatter(
	    x=x+x_rev,
	    y=y_upper+y_lower,
	    fill='tozerox',
	    fillcolor='rgba('+str(r)+','+str(g)+','+str(b)+',0.2)',
	    line=Line(color='transparent'),
	    # name='Premium',
	    showlegend=False,
	)

	trace2 = Scatter(
	    x=x,
	    y=y,
	    line=Line(color='rgba('+str(r)+','+str(g)+','+str(b)+')'),
	    mode='lines',
	    name=name,
	)
	return trace1,trace2

if False:
	dqn_trace1,dqn_trace2=get_reward_plot_synthetic('Baseline', 'logs/dqn_synthetic',0,0,0)
	trace1,trace2=get_reward_plot_synthetic('Our Approach', 'logs/synthetic',0,176,246)

	data = Data([dqn_trace2, dqn_trace1 , trace1,trace2])
	fig = Figure(data=data, layout=get_layout())
	plotly.offline.plot(fig, filename='Reward')

############# plotting subgoals ############

def get_plots_realexp_subgoals(goalid=6):
	f = h5py.File('stats.h5', "r")
	global total_goals 
	data_db=dict();cnt=0
	for exp in glob.glob('logs/latest/*'):
		data = np.loadtxt(exp+'/test_avgR.log')
		data_db[cnt]=data
		cnt=cnt+1

	num_exps = len(data_db)
	print(num_exps)

	desc = ['top-left-door', 'top-right-door', 'middle-ladder', 'bottom-left-ladder', 'bottom-right-ladder', 'key']

	if True:# for i in range(1,total_goals+1):
		i = goalid
		hitrate_agg = dict()
		cnt=0
		for expid in range(1,num_exps+1):
			# expid=3
			# print(expid)
			hitrate = f['run' + str(expid) + '_subgoal_hitrate_gid_'+str(i)][:]
			itrs = f['run' + str(expid) + '_subgoal_itr_gid_'+str(i)][:]
			sindxs = np.argsort(itrs)
			itrs = itrs[sindxs]
			hitrate = hitrate[sindxs]

			hitrate_basic = f['pretrain_subgoal_hitrate_gid_'+str(i)][:]
			itrs_basic = f['pretrain_subgoal_itr_gid_'+str(i)][:]
			sindxs_basic = np.argsort(itrs_basic)
			itrs_basic = itrs_basic[sindxs_basic]
			hitrate_basic = hitrate_basic[sindxs_basic]

			# itrs = itrs + 2300000 #offset from basic
			# itrs = np.hstack((itrs_basic, itrs))
			# hitrate = np.hstack((hitrate_basic, hitrate))
			
			itrs_dic=dict()
			hitrate_map=dict()
			for j in range(np.shape(itrs)[0]):
				itrs_dic[itrs[j]]=True
				hitrate_map[itrs[j]] = hitrate[j]

			hitrate_expanded = []
			itrs_expanded = []
			for j in range(10000,np.int64(itrs[-1]),10000):
				if j in itrs_dic:
					hitrate_expanded += [hitrate_map[j]]
				else:
					hitrate_expanded += [0]
				itrs_expanded += [j]
			hitrate = hitrate_expanded
			itrs = itrs_expanded
			hitrate_agg[cnt] = hitrate
			cnt=cnt+1
			# print(itrs_expanded)

		minlen=np.inf
		for k in range(len(hitrate_agg)):
			minlen = min(len(hitrate_agg[k]), minlen)

		y=np.zeros(minlen)
		agg=np.zeros((len(hitrate_agg),minlen))
		for k in range(len(hitrate_agg)):
			y = y + hitrate_agg[k][0:minlen]
			agg[k,:] = hitrate_agg[k][0:minlen]
		y = y/len(hitrate_agg)
		x = itrs[0:minlen]
		std = np.std(agg,axis=0)
		y_lower = y-std
		y_lower=y_lower[::-1]
		y_upper = y+std
		x_rev = x[::-1]
		x_rev=list(x_rev);y_lower=list(y_lower);y_upper=list(y_upper)
		# pdb.set_trace()

		x=list(x); y=list(y);


		trace1 = Scatter(
		    x=x+x_rev,
		    y=y_upper+y_lower,
		    fill='tozerox',
		    fillcolor='rgba(0,176,246,0.2)',
		    line=Line(color='transparent'),
		    # name='Premium',
		    showlegend=False,
		)

		trace2 = Scatter(
		    x=x,
		    y=y,
		    line=Line(color='rgb(0,176,246)'),
		    mode='lines',
		    # name='Premium',
		)

		data = Data([trace1, trace2])
		fig = Figure(data=data, layout=get_layout(ytitle='Average Goal Success Rate'))
		plotly.offline.plot(fig, filename='subgoal_'+str(i))





def get_plots_realexp_avgsubgoals():
	f = h5py.File('stats.h5', "r")
	global total_goals 
	data_db=dict();cnt=0
	for exp in glob.glob('logs/latest/*'):
		data = np.loadtxt(exp+'/test_avgR.log')
		data_db[cnt]=data
		cnt=cnt+1

	num_exps = len(data_db)
	print(num_exps)

	desc = ['top-left-door', 'top-right-door', 'middle-ladder', 'bottom-left-ladder', 'bottom-right-ladder', 'key']

	hitrate_agg = dict()
	chosensubg_agg = dict()
	cnt=0
	final_chosensubg = dict()
	for i in range(1,total_goals+1):
		for expid in range(1,num_exps+1):
			# expid=3
			# print(expid)
			hitrate = f['run' + str(expid) + '_subgoal_hitrate_gid_'+str(i)][:]
			itrs = f['run' + str(expid) + '_subgoal_itr_gid_'+str(i)][:]
			choseng = f['run' + str(expid) + '_subgoal_total_gid_'+str(i)][:]
			
			sindxs = np.argsort(itrs)
			itrs = itrs[sindxs]
			itrs_original = copy.deepcopy(itrs)
			hitrate = hitrate[sindxs]
			choseng = choseng[sindxs]
			# hitrate = np.clip(hitrate,0,1)

			hitrate_basic = f['pretrain_subgoal_hitrate_gid_'+str(i)][:]
			itrs_basic = f['pretrain_subgoal_itr_gid_'+str(i)][:]
			sindxs_basic = np.argsort(itrs_basic)
			itrs_basic = itrs_basic[sindxs_basic]
			hitrate_basic = hitrate_basic[sindxs_basic]
			# hitrate_basic = np.clip(hitrate_basic,0,1)

			# itrs = itrs + 2300000 #offset from basic
			# itrs = np.hstack((itrs_basic, itrs))
			# hitrate = np.hstack((hitrate_basic, hitrate))
			
			itrs_dic=dict()
			hitrate_map=dict(); choseng_map = dict()
			for j in range(np.shape(itrs)[0]):
				itrs_dic[itrs[j]]=True
				hitrate_map[itrs[j]] = hitrate[j]
				choseng_map[itrs[j]] = choseng[j]

			hitrate_expanded = []; choseng_expanded = [];
			itrs_expanded = []
			for j in range(10000,np.int64(itrs[-1]),10000):
				if j in itrs_dic:
					hitrate_expanded += [hitrate_map[j]]
					choseng_expanded += [choseng_map[j]]
				else:
					hitrate_expanded += [0]
					choseng_expanded += [0]
				itrs_expanded += [j]

			hitrate = hitrate_expanded
			itrs = itrs_expanded
			choseng = choseng_expanded

			hitrate_agg[cnt] = hitrate
			chosensubg_agg[cnt] = choseng
			cnt=cnt+1
			# print(itrs_expanded)

		minlen_chosen = np.inf
		for k in range(len(chosensubg_agg)):
			minlen_chosen =  min(len(chosensubg_agg[k]), minlen_chosen)
		agg_chosen = np.zeros((len(chosensubg_agg),minlen_chosen))
		for k in range(len(hitrate_agg)):
			agg_chosen[k,:] = chosensubg_agg[k][0:minlen_chosen]
		final_chosensubg[i-1] = np.mean(agg_chosen,0)

	minlen=np.inf; 
	for k in range(len(hitrate_agg)):
		minlen = min(len(hitrate_agg[k]), minlen)
	
	y=np.zeros(minlen)
	agg=np.zeros((len(hitrate_agg),minlen))
	for k in range(len(hitrate_agg)):
		y = y + hitrate_agg[k][0:minlen]
		agg[k,:] = hitrate_agg[k][0:minlen]

	y = y/len(hitrate_agg)
	x = itrs[0:minlen]
	std = np.std(agg,axis=0)
	y_lower = y-std
	y_lower=y_lower[::-1]
	y_upper = y+std
	x_rev = x[::-1]
	x_rev=list(x_rev);y_lower=list(y_lower);y_upper=list(y_upper)
	x=list(x); y=list(y);

	trace1 = Scatter(
	    x=x+x_rev,
	    y=y_upper+y_lower,
	    fill='tozerox',
	    fillcolor='rgba(0,176,246,0.2)',
	    line=Line(color='transparent'),
	    # name='Premium',
	    showlegend=False,
	)

	trace2 = Scatter(
	    x=x,
	    y=y,
	    line=Line(color='rgb(0,176,246)'),
	    mode='lines',
	    # name='Premium',
	)

	data = Data([trace1, trace2])
	fig = Figure(data=data, layout=get_layout(ytitle='Average Goal Success Rate'))
	plotly.offline.plot(fig, filename='Average ratio across sub-goals')


	# normalize values
	for j in [50,100,150,200]:
		Z = 0
		for k in range(len(final_chosensubg)):
			Z += final_chosensubg[k][j] 
		for k in range(len(final_chosensubg)):
			final_chosensubg[k][j] = final_chosensubg[k][j] / Z

	###### bar graph ######
	traces=[]
	colors = ['rgb(189,130,49)', 'rgb(130,189,49)', 'rgb(189,49, 130)', 'rgb(130,49, 189)', 'rgb(189,0,49)', 'rgb(0,130,49)']
	names = ['top-left door', 'top-right door', 'middle-ladder', 'bottom-left-ladder', 'bottom-right-ladder', 'key	']
	for j in range(len(final_chosensubg)):

		trace = go.Bar(
		    x=['0.5M', '1M', '1.5M', '2M'],
		    y=[final_chosensubg[j][50],final_chosensubg[j][100], final_chosensubg[j][150], final_chosensubg[j][200]],
		    name=names[j],
		    marker=dict(
		        color=colors[j],
		    )
		)
		traces += [trace]
	data = Data(traces)
	fig = Figure(data=data, layout=get_layout('Steps', '% goal chosen')) 
	plotly.offline.plot(fig, filename='Bar graph')



get_reward_plot_realexp()
# get_plots_realexp_subgoals(6)
# get_plots_realexp_avgsubgoals()


