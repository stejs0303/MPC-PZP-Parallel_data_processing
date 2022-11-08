from collected_data import Collected_data
import matplotlib.pyplot as plt
import numpy as np

#TODO: Fix later

def show_graph(*collected_data_list: Collected_data):
    names: list = []
    sub_names: list = ["processing", "other",
                       "processing", "data preparation", "other",
                       "memory manipulation", "processing", "data preparation", "other"]
    execution_times: list = []
    sub_times: list = []
    sum: int = 0

    for collected_data in collected_data_list:
        names.append(f"{collected_data.type}, threads: {collected_data.threads}")
        
        if collected_data.memory_manipulation_time != 0:
            sub_times.append(collected_data.memory_manipulation_time)
    
        sub_times.append(collected_data.processing_time)
        
        if collected_data.preparation_time != 0:
            sub_times.append(collected_data.preparation_time)
        
        sub_times.append(collected_data.execution_time - collected_data.processing_time - 
                     collected_data.preparation_time - collected_data.memory_manipulation_time)
            
        execution_times.append(collected_data.execution_time)
    
    plt.rcParams["figure.figsize"] = [9.50, 6]
    plt.rcParams["figure.autolayout"] = True
    
    _, ax = plt.subplots()

    size = 0.3
   
    cmap = plt.get_cmap("tab20c")
    inner_colors = cmap(np.arange(3)*4)
    outer_colors = cmap(np.array([1, 19, 5, 6, 19, 11, 9, 10, 19]))

    wedges, _ = ax.pie(sub_times,  radius=1, colors=outer_colors,
           wedgeprops=dict(width=size, edgecolor='w'))
    
    ax.pie(execution_times, labels=names, radius=1-size, colors=inner_colors,
           wedgeprops=dict(width=size, edgecolor='w'))
    
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(sub_names[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)
    
    ax.set(aspect="equal", title='Pie plot with `ax.pie`')
    
    plt.show()
        
    if __name__=="__main__":
        show_graph()