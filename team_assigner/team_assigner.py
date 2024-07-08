
from sklearn.cluster import KMeans
class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None
        pass
    
    def get_cluster_model(self,image):
        #use kmeans to cluster the image into 2 color, usually can identify the player and the background
        image_2d = image.reshape(-1,3)
        kmeans = KMeans(n_clusters=2,init="k-means++",n_init=1)
        kmeans.fit(image_2d)
        return kmeans
    
    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        
        #get the top half of the image for better cluster 
        top_half_img = image[:int(image.shape[0]/2),:]
        
        kmeans = self.get_cluster_model(top_half_img)
        
        labels = kmeans.labels_
        
        clustered_img = labels.reshape(top_half_img.shape[0],top_half_img.shape[1])
        
        # the corner often have the same color with the background
        corner_clusters = [clustered_img[0,0],clustered_img[0,-1],clustered_img[-1,0],clustered_img[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key = corner_clusters.count) # get the most common cluster in the corner, its the background
        non_player_cluster = 1 - non_player_cluster # get the other cluster, which is the player
        
        player_color = kmeans.cluster_centers_[non_player_cluster]
        
        return player_color
            
    def assign_team_color(self,frame,player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["boxx"]
            player_color = self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        # use kmeans to cluster the color of players into 2 cluster    
        kmeans = KMeans(n_clusters=2,init="k-means++",n_init=1)
        kmeans.fit(player_colors)
        
        self.kmeans = kmeans
        
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        
    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        if self.kmeans is None:
            raise ValueError("Please assign team color first")
        
        player_color = self.get_player_color(frame,player_bbox)
        
        # predict the team of the player based on color
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id = team_id + 1
        
        # assign the team to the player
        self.player_team_dict[player_id] = team_id
        
        return team_id
        
    
        