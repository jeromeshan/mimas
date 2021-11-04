import plotly.graph_objects as go
def go_cluster(cluster):
    
    if(len(cluster.galaxies)==1):
        return [go.Scatter3d(
    x=cluster.galaxies[:,0],
    y=cluster.galaxies[:,1],
    z=cluster.galaxies[:,2],
    mode='markers',
    marker=dict(
        size=1,
        opacity=1,
        color='green'
    )
    )]
    
    return [go.Scatter3d(
    x=cluster.galaxies[:,0],
    y=cluster.galaxies[:,1],
    z=cluster.galaxies[:,2],
    mode='markers',
    marker=dict(
        size=3,
        opacity=1,
        color='green'

    )
    )]+[get_mesh(cluster)]+[go.Scatter3d(
    x=[cluster.centroid[0]],
    y=[cluster.centroid[1]],
    z=[cluster.centroid[2]],
    mode='markers',
    marker=dict(
        size=2,
        opacity=1,
        color='red'
    )
    )]

def get_mesh(self):
        return go.Mesh3d(
        x=self.rotated_cube[:,0],
        y=self.rotated_cube[:,1],
        z=self.rotated_cube[:,2],
        opacity=0.5,
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],color='blue'
        )

def get_clusters_plot(clusters,is_show=False,filename='3d_model'):
    
    data=[]
    for cluster in clusters:
        data+=go_cluster(cluster)
    
    fig = go.Figure(data=data)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    if(is_show):
        fig.show()
    fig.write_html(filename+".html")