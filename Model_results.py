# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
import numpy as np
from scipy.stats import zscore
import scipy.stats as stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import PIL.Image as Image
import torchvision
import torchvision.transforms as transforms
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('D:\\TDCNN')
import BrainSOM
import Hopfield_VTCSOM


### Data
data_transforms = {
    'see': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
    }
        
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()


def cohen_d(x1, x2):
    s1 = x1.std()
    return (x1.mean()-x2)/s1

def Functional_map_pca(som, pca, pca_index): 
    class_name = ['face', 'place', 'body', 'object']
    f1 = os.listdir(".\HCP_WM\\" + 'face')
    if '.DS_Store' in f1:
        f1.remove('.DS_Store')
    f2 = os.listdir(".\HCP_WM\\" + 'place')
    if '.DS_Store' in f2:
        f2.remove('.DS_Store')
    f3 = os.listdir(".\HCP_WM\\" + 'body')
    if '.DS_Store' in f3:
        f3.remove('.DS_Store')
    f4 = os.listdir(".\HCP_WM\\" + 'object')
    if '.DS_Store' in f4:
        f4.remove('.DS_Store')
    Response = []
    for index,f in enumerate([f1,f2,f3,f4]):
        for pic in f:
            img = Image.open(".\HCP_WM\\"+class_name[index]+"\\"+pic).convert('RGB')
            picimg = data_transforms['val'](img).unsqueeze(0) 
            output = alexnet(picimg).data.numpy()
            Response.append(output[0])
    Response = np.array(Response) 
    mean_features = np.mean(Response, axis=0)
    std_features = np.std(Response, axis=0)
    Response = zscore(Response, axis=0)
    Response_som = []
    for response in Response:
        Response_som.append(1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index]))
    Response_som = np.array(Response_som)
    return Response_som, (mean_features, std_features)

def som_mask(som, Response, Contrast_respense, contrast_index, threshold_cohend):
    t_map, p_map = stats.ttest_1samp(Response, Contrast_respense[contrast_index])
    mask = np.zeros((som._weights.shape[0],som._weights.shape[1])) - 1
    Cohend = []
    for i in range(som._weights.shape[0]):
        for j in range(som._weights.shape[1]):
            cohend = cohen_d(Response[:,i,j], Contrast_respense[contrast_index][i,j])
            Cohend.append(cohend)
            if (p_map[i,j] < 0.05/40000) and (cohend>threshold_cohend):
                mask[i,j] = 1
    return mask  

def Pure_picture_activation(pic_dir, prepro_method, som, pca, pca_index, mean_features, std_features):
    img = Image.open(pic_dir).convert('RGB')
    if prepro_method=='val':
        picimg = data_transforms['val'](img)
        img_see = np.array(data_transforms['see'](img))
    picimg = picimg.unsqueeze(0) 
    output = alexnet(picimg).data.numpy()
    response = (output-mean_features)/std_features
    response_som = 1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index])
    return img_see, output, response_som
                    
                
                
### sigma=6.2
som = BrainSOM.VTCSOM(200, 200, 4, sigma=6.2, learning_rate=1, neighborhood_function='gaussian')
som._weights = np.load('./som_sigma_6.2.npy')

Data = np.load('.\\Data.npy')
Data = zscore(Data)
pca = PCA()
pca.fit(Data)
Response_som, (mean_features,std_features) = Functional_map_pca(som, pca, [0,1,2,3])
Response_face = Response_som[:111,:,:]
Response_place = Response_som[111:172,:,:]
Response_body = Response_som[172:250,:,:]
Response_object = Response_som[250:,:,:]
Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
threshold_cohend = 0.5
face_mask = som_mask(som, Response_face, Contrast_respense, 0, threshold_cohend)
place_mask = som_mask(som, Response_place, Contrast_respense, 1, threshold_cohend)
limb_mask = som_mask(som, Response_body, Contrast_respense, 2, threshold_cohend)
object_mask = som_mask(som, Response_object, Contrast_respense, 3, threshold_cohend)
training_pattern = np.array([face_mask.reshape(-1),
                             place_mask.reshape(-1),
                             limb_mask.reshape(-1),
                             object_mask.reshape(-1)])


model = Hopfield_VTCSOM.Stochastic_Hopfield_nn(x=200, y=200, pflag=1, nflag=-1,
                                               patterns=[face_mask,place_mask,limb_mask,object_mask])
model.reconstruct_w_with_structure_constrain([training_pattern], 'exponential', 0.023)





"Specialization vs Generality"
###############################################################################
###############################################################################
def Get_mean_std(): 
    class_name = ['face', 'place', 'body', 'object']
    f1 = os.listdir(".\HCP_WM\\" + 'face')
    if '.DS_Store' in f1:
        f1.remove('.DS_Store')
    f2 = os.listdir(".\HCP_WM\\" + 'place')
    if '.DS_Store' in f2:
        f2.remove('.DS_Store')
    f3 = os.listdir(".\HCP_WM\\" + 'body')
    if '.DS_Store' in f3:
        f3.remove('.DS_Store')
    f4 = os.listdir(".\HCP_WM\\" + 'object')
    if '.DS_Store' in f4:
        f4.remove('.DS_Store')
    Response = []
    for index,f in enumerate([f1,f2,f3,f4]):
        for pic in f:
            img = Image.open(".\HCP_WM\\"+class_name[index]+"\\"+pic).convert('RGB')
            picimg = data_transforms['val'](img).unsqueeze(0) 
            output = alexnet(picimg).data.numpy()
            Response.append(output[0])
    Response = np.array(Response) 
    return Response.mean(axis=0), Response.std(axis=0)
mean, std = Get_mean_std()

def Recurrent_results(stim_file_path, mean,std):
    files = os.listdir(stim_file_path)
    Image_state_dict = dict()
    External_field_prior = np.zeros((200,200))
    Dynamic_states = []
    for index,f in enumerate(files):
        print(index)
        model.rebuild_up_param()
        pic_dir = stim_file_path + f
        img_see, img_mask_see, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean, std)
        initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
        stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                                 H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                 epochs=150000, save_inter_step=1000)
        Dynamic_states.append(model.dynamics_state)
        Image_state_dict[index] = stable_state[0].reshape(200,200)
        print(index, stable_state[0].reshape(200,200)[np.where(face_mask==1)[0], np.where(face_mask==1)[1]].mean())
    images_response = []
    for k,v in Image_state_dict.items():
        images_response.append(v)
    images_response = np.array(images_response)
    Dynamic_states = np.array(Dynamic_states)
    return images_response, Dynamic_states


### Specialization
images_response,Dynamic_states = Recurrent_results('all_stim\\', mean,std)
### Generality
images_response,Dynamic_states = Recurrent_results('ge_stim\\', mean,std)
images_response,Dynamic_states = Recurrent_results('f4n\\', mean,std)




def plot_se_stable_state_bar(Data, order_list, legend_list, color_list):
    def stats_value(x):
        timeseries = x[:,np.where(face_mask==1)[0], np.where(face_mask==1)[1]].mean(axis=1)
        return timeseries
    # plot
    images_response_mean = []
    images_response_std = []
    for index,i in enumerate(order_list):
        data = Data[i]
        images_response = stats_value(data)
        images_response_mean.append(images_response.mean())
        images_response_std.append(images_response.std()/np.sqrt(images_response.shape[0]))
    plt.figure(figsize=(4,4), dpi=400)
    plt.bar(range(10), images_response_mean, color=color_list, alpha=0.8)
    plt.errorbar(range(10), images_response_mean, 
                yerr=images_response_std,
                fmt='.', ecolor='black',elinewidth=2,capsize=4)
    plt.xticks(range(10),legend_list, rotation=70, fontsize=10)
    plt.yticks([-1,0,1], fontsize=10)
    plt.ylabel('Activation', fontsize=15)

def plot_ge_stable_state_bar(Data, order_list, legend_list, color_list):
    def stats_value(x):
        timeseries = x[:,np.where(face_mask==1)[0], np.where(face_mask==1)[1]].mean(axis=1)
        return timeseries
    # plot
    images_response_mean = []
    images_response_std = []
    for index,i in enumerate(order_list):
        data = Data[i]
        images_response = stats_value(data)
        images_response_mean.append(images_response.mean())
        images_response_std.append(images_response.std()/np.sqrt(images_response.shape[0]))
    plt.figure(figsize=(4,4), dpi=300)
    plt.bar(range(8), images_response_mean, color=color_list, alpha=0.7)
    plt.errorbar(range(8), images_response_mean, 
                yerr=images_response_std,
                fmt='.', ecolor='black',elinewidth=2,capsize=4)
    plt.xticks(range(8),legend_list, rotation=70, fontsize=10)
    plt.yticks([-1,0,1], fontsize=10)
    plt.ylabel('Activation', fontsize=15)                             
    


# Bar Figure
lamda = '0.023'
dir_name = './lamda_' + lamda + '\\'
images_response_path_list = [dir_name+'se_recurrent_images_response.npy',
                             dir_name+'ge_recurrent_images_response.npy',
                             dir_name+'f4n_recurrent_images_response.npy']

images_response = np.load(images_response_path_list[0])
Data = []
for i in np.arange(0,200,20):
    Data.append(images_response[i:i+20])
plot_se_stable_state_bar(Data,
                          order_list=[4,2,3,5,0,9,1,6,7,8],
                          legend_list=['face','cat','dog','lemon','ambulance','store','backpack','pitcher','plane','speaker'],
                          color_list=['red','darkred','indianred','orange','grey','deepskyblue','royalblue','blue','darkblue','black'])


images_response = np.load(images_response_path_list[1])
Data = []
Data.append(images_response[:10])
Data.append(images_response[10:20])
Data.append(images_response[20:30])
Data.append(images_response[55:65])
Data.append(images_response[35:45])
images_response = np.load(images_response_path_list[2])
for i in np.arange(0,75,15):
    Data.append(images_response[i:i+15])
plot_ge_stable_state_bar(Data,
                         order_list=[0,1,3,5,6,7,8,9],
                         legend_list=['cat face','dog face','tiger face','front','profile','cheek','back','tool'],
                         color_list=['bisque','gold','yellow','red','darkred','indianred','orange','darkblue'])







"Energy"
###############################################################################
###############################################################################
from matplotlib.colors import LinearSegmentedColormap
from scipy import interpolate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clist=['Black','Grey']
newcmap = LinearSegmentedColormap.from_list('chaos',clist)


class classifier(torch.nn.Module):
    def __init__(self, layer1_in, layer1_out, layer2_in, layer2_out, layer3_in, layer3_out, max_iter):
        super(classifier, self).__init__()
        self.max_iter = max_iter
        self.net = torch.nn.Sequential(torch.nn.Linear(layer1_in, layer1_out, bias=True),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(layer2_in, layer2_out, bias=True),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(layer3_in, layer3_out, bias=True)).to(device) 
    def fit(self, X, TARGETS):
        LOSS_test = []
        temp = np.arange(0,16000,1)
        np.random.shuffle(temp)
        index_train = temp[:15000]
        index_test = temp[15000:]
        X = torch.tensor(X,dtype=torch.float32).to(device) 
        TARGETS = torch.tensor(TARGETS,dtype=torch.float32).to(device) 
        X_train = X[index_train]
        TARGETS_train = TARGETS[index_train]
        X_test = X[index_test]
        TARGETS_test = TARGETS[index_test]
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        loss_function = torch.nn.MSELoss()
        for epoch in range(self.max_iter):  
            out = self.net.forward(X_train)  
            loss = loss_function(out, TARGETS_train) 
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()
            out_test = self.net.forward(X_test)  
            loss_test = loss_function(out_test, TARGETS_test) 
            LOSS_test.append(loss_test.item())
        return LOSS_test, out_test.cpu().data.numpy(), TARGETS_test.cpu().data.numpy()      

def X_and_H(lamda):
    # 1000 stimuli state    
    H = np.load(r'.\1000_H_'+lamda+'.npy')    
    return H

def X_and_H_se_ge(lamda, method='recurrent', state='stable_state'):
    # X se and ge
    dir_name = r'.\lamda_' + lamda + '\\'
    if method=='recurrent':
        if state=='stable_state':
            images_response_path_list = [dir_name+'se_recurrent_images_response.npy',
                                             dir_name+'ge_recurrent_images_response.npy',
                                             dir_name+'f4n_recurrent_images_response.npy']
        if state=='dynamical_state':
            images_response_path_list = [dir_name+'se_recurrent_Dynamic_states.npy',
                                             dir_name+'ge_recurrent_Dynamic_states.npy',
                                             dir_name+'f4n_recurrent_Dynamic_states.npy']
    if method=='feedback':
        if state=='stable_state':
            images_response_path_list = [dir_name+'se_face_feedback_images_response.npy',
                                             dir_name+'ge_object_feedback_images_response.npy',
                                             dir_name+'f4n_object_feedback_images_response.npy']
        if state=='dynamical_state':
            images_response_path_list = [dir_name+'se_face_feedback_Dynamic_states.npy',
                                             dir_name+'ge_object_feedback_Dynamic_states.npy',
                                             dir_name+'f4n_object_feedback_Dynamic_states.npy']
    if state=='stable_state':
        Data = np.load(images_response_path_list[0])
        for i in np.arange(0,200,20):
            if i==0:
                images_response_se = Data[i:i+20].reshape(20,-1)
            else:
                images_response_se = np.vstack((images_response_se, Data[i:i+20].reshape(20,-1)))
        images_response_se = [images_response_se[i*20:i*20+20] for i in [4,2,3,5,0,9,1,6,7,8]]        
        images_response_ge = []
        Data = np.load(images_response_path_list[1])
        images_response_ge.append(Data[:10].reshape(10,-1))
        images_response_ge.append(Data[10:20].reshape(10,-1))
        images_response_ge.append(Data[55:65].reshape(10,-1))
        Data = np.load(images_response_path_list[2])
        for i in np.arange(0,75,15):
            images_response_ge.append(Data[i:i+15].reshape(15,-1))    
    
    if state=='dynamical_state':
        Data = np.load(images_response_path_list[0])
        Data = Data[:,np.arange(0,Data.shape[1],10),:,:]
        for i in np.arange(0,200,20):
            if i==0:
                images_response_se = Data[i:i+20].reshape(20,Data.shape[1],-1)
            else:
                images_response_se = np.vstack((images_response_se, Data[i:i+20].reshape(20,Data.shape[1],-1)))
        images_response_se = [images_response_se[i*20:i*20+20] for i in [4,2,3,5,0,9,1,6,7,8]]
        images_response_ge = []
        Data = np.load(images_response_path_list[1])
        Data = Data[:,np.arange(0,Data.shape[1],10),:,:]
        images_response_ge.append(Data[:10].reshape(10,Data.shape[1],-1))
        images_response_ge.append(Data[10:20].reshape(10,Data.shape[1],-1))
        images_response_ge.append(Data[55:65].reshape(10,Data.shape[1],-1))
        Data = np.load(images_response_path_list[2])
        Data = Data[:,np.arange(0,Data.shape[1],10),:,:]
        for i in np.arange(0,75,15):
            images_response_ge.append(Data[i:i+15].reshape(15,Data.shape[1],-1))  
        
    # H se and ge
    if state=='stable_state':
        if method=='recurrent':
            H_temp = np.load(dir_name + 'H_se_ge.npy')
        if method=='feedback':
            H_temp = np.load(dir_name + 'H_se_face_feedback_ge_object_feedback.npy')
    if state=='dynamical_state':
        if method=='recurrent':
            H_temp = np.load(dir_name + 'H_se_ge_dynamics.npy')
        if method=='feedback':
            H_temp = np.load(dir_name + 'H_se_face_feedback_ge_object_feedback_dynamics.npy')
    H_se = [H_temp[i*20:i*20+20] for i in [4,2,3,5,0,9,1,6,7,8]]
    H_ge = []
    H_ge.append(H_temp[200:210])
    H_ge.append(H_temp[210:220])
    H_ge.append(H_temp[230:240])
    H_ge.append(H_temp[250:265])
    H_ge.append(H_temp[265:280])
    H_ge.append(H_temp[280:295])
    H_ge.append(H_temp[295:310])
    H_ge.append(H_temp[310:325])
    return images_response_se, images_response_ge, H_se, H_ge     

def plot_surface(X_embedded, H, e, plot_points=False, plot_face=False):
    def fake_point(X_embedded,p,q):
        X_fake = []
        np.random.seed(0)
        for i in np.random.uniform(p,q,(100,2)):
            d_min = np.linalg.norm(X_embedded-i, axis=1).min()
            if d_min>3:
                X_fake.append(i)
        X_fake = np.array(X_fake)
        H_fake = np.zeros(X_fake.shape[0])
        return X_fake, H_fake
    def transfer_H(H):
        H = np.where(H>0, 0, H)
        H = -np.power(-H, 0.4)    
        return H
    def low_pass_filter(znew, width):
        f = np.fft.fft2(znew) 
        fshift = np.fft.fftshift(f)
        rows, cols = znew.shape
        crow, ccol = int(rows/2) , int(cols/2)     # 中心位置        
        # 低通滤波
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow-width:crow+width, ccol-width:ccol+width] = 1
        # IDFT
        fshift = fshift*mask
        f_ishift = np.fft.ifftshift(fshift)
        znew_back = np.fft.ifft2(f_ishift)
        znew_back = -np.abs(znew_back)
        return znew_back
    # log(H)
    H_ = transfer_H(H)
    min_value = -100
    # generate fake point
    a,b = [np.floor(X_embedded[:,0].min()-e),np.ceil(X_embedded[:,0].max()+e)]
    c,d = [np.floor(X_embedded[:,1].min()-e),np.ceil(X_embedded[:,1].max()+e)]
    X_fake, H_fake = fake_point(X_embedded,p=min(a,c),q=max(b,d))
    # surface fit
    func = interpolate.Rbf(np.hstack((X_embedded[:,0],X_fake[:,0])),
                           np.hstack((X_embedded[:,1],X_fake[:,1])),
                           np.hstack((H_,H_fake)), function='multiquadric', smooth=1)
    xnew,ynew = np.mgrid[a:b:0.2,c:d:0.2]
    znew = func(xnew,ynew)
    znew_back = low_pass_filter(znew, width=8)      
    return xnew,ynew,znew_back

def points_in_plane(X_new_list, label_list, color_list):
    plt.figure(figsize=(5,5), dpi=300)
    plt.style.use('seaborn')
    for index,x_new in enumerate(X_new_list):
        plt.scatter(x_new[:,:,0],x_new[:,:,1], s=10, 
                    c=color_list[index], alpha=np.linspace(0.2,1,16))
        plt.scatter(x_new[:,-1,0],x_new[:,-1,1], s=20, 
                    label=label_list[index], c=color_list[index], alpha=1)
    #plt.legend(loc='lower left', fontsize=10)
    plt.axis('off')
    
def points_in_3D(X_new_list, H_new_list, label_list, color_list):
    def transfer_H(H):
        H = np.where(H>0, 0, H)
        H = -np.power(-H, 0.4)    
        return H
    fig = plt.figure(figsize=(5,5), dpi=300)
    plt.style.use('seaborn')
    ax = Axes3D(fig)
    for index,x_new in enumerate(X_new_list):
        for step in range(16):
            ax.scatter(x_new[:,step,0],x_new[:,step,1],transfer_H(H_new_list[index][:,step]), s=10, 
                        c=color_list[index], alpha=(step+5)/20)
        ax.scatter(x_new[:,-1,0],x_new[:,-1,1],transfer_H(H_new_list[index][:,-1]), s=20, 
                    label=label_list[index], c=color_list[index], alpha=1)
    ax.patch.set_facecolor("white")
    plt.axis('off')

def points_to_surface(xnew,ynew,znew, X_new_list, color_list=['red','darkred','gold','orange','darkblue','black']):
    cate_index = {'face':[0],
                 'species':[1,2,10,11,12],
                 'view-point':[13,14,15,16],
                 'shape-similar':[3],
                 'structure-similar':[4,8],
                 'familarity-control':[5,6,7,9,17]}
    # plot surface
    min_value = -100
    fig = plt.figure(figsize=(5,5), dpi=300)
    plt.style.use('seaborn')
    ax = Axes3D(fig)
    ax.plot_surface(xnew,ynew,znew, cmap='YlGn_r', alpha=0.7,rstride=5, cstride=5)
    ax.contour(xnew,ynew,znew, zdir='z',
               levels=[znew.min()*0.9, znew.min()*0.8, znew.min()*0.7], 
               offset=min_value, cmap=newcmap, linewidths=1)
    # plot dots
    for i,(keys,values) in enumerate(cate_index.items()):
        x_new = [X_new_list[v] for v in values]
        x_new_center = np.vstack([X_new_list[v] for v in values]).mean(0)
        for x in x_new: 
            x = x.mean(0)
            ax.scatter(x[0],x[1], min_value, s=100, c=color_list[i], alpha=1)
        ax.scatter(x_new_center[0],x_new_center[1], min_value, s=500, c=color_list[i], alpha=1, label=keys)
    ax.set_zlim([min_value,0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.patch.set_facecolor("white")
    plt.show()
        
def path_length(x):
    temp = []
    for index in range(x.shape[0]-1):
        temp.append(np.sqrt(np.sum((x[index]-x[index+1])**2)))
    return sum(temp)



lamda = '0.023'
H = X_and_H(lamda=lamda)
H = H.reshape(16000)

## Paramatic-UMAP
X_embedded = np.load('.\X_embedded_'+lamda+'_UMAP.npy')
net_umap = torch.load('.\Paramatic_net_'+lamda)


## 1000 stimuli Path length
average_path_length_0 = np.load(r'.\average_path_length_0.0.npy')
average_path_length_023 = np.load(r'.\average_path_length_0.023.npy')
average_path_length_1 = np.load(r'.\average_path_length_0.1.npy')

plt.figure(figsize=(4,6), dpi=300)
plt.style.use('seaborn')
font_1 = {"size": 15}
sns.barplot(data=[average_path_length_0, average_path_length_023, average_path_length_1])
plt.xlabel("Interconnectivity", font_1)
plt.ylabel("Average path length", font_1)
plt.xticks(ticks = [0, 1, 2], fontsize = 11)
plt.yticks(fontsize=12)



## Energy Representational manifold
X_se_recurrent, X_ge_recurrent, H_se_recurrent, H_ge_recurrent = X_and_H_se_ge(lamda=lamda, method='recurrent')
outputs_se_recurrent = []
for x_se_recurrent in X_se_recurrent:
    outputs_se_recurrent.append(net_umap.net.forward(torch.tensor(x_se_recurrent,dtype=torch.float32).to(device)).cpu().data.numpy())
outputs_ge_recurrent = []
for x_ge_recurrent in X_ge_recurrent:
    outputs_ge_recurrent.append(net_umap.net.forward(torch.tensor(x_ge_recurrent,dtype=torch.float32).to(device)).cpu().data.numpy())

xnew,ynew,znew = plot_surface(X_embedded, H, e=5)
X_new_list = []
for i in outputs_se_recurrent:
    X_new_list.append(i)
for i in outputs_ge_recurrent:
    X_new_list.append(i)
points_to_surface(xnew,ynew,znew,X_new_list)







"""Cognitive impenetrability"""
###############################################################################
def plot_condition_timeseries(Data, order_list, legend_list, color_list, label_colation=None):
    def stats_value(x):
        timeseries = x[:,:,np.where(face_mask==1)[0], np.where(face_mask==1)[1]].mean(axis=2)
        return timeseries
    # plot
    plt.figure(figsize=(4,8),dpi=100)
    plt.style.use('seaborn')
    for index,i in enumerate(order_list):
        data = Data[i]
        images_response = stats_value(data)
        plt.plot(range(images_response.shape[1]), 
                 images_response.mean(0),
                 label=legend_list[index], c=color_list[index])
        plt.fill_between(range(images_response.shape[1]),
                         images_response.mean(0)-(images_response.std(0)/np.sqrt(images_response.shape[0])),
                         images_response.mean(0)+(images_response.std(0)/np.sqrt(images_response.shape[0])),
                         alpha=0.3)
    plt.ylim([-1,1])
    plt.legend()
    if label_colation!=None:
        plt.legend(bbox_to_anchor=label_colation)
        
def plot_se_stable_state_bar(Data, order_list, legend_list, color_list):
    def stats_value(x):
        timeseries = x[:,:,np.where(face_mask==1)[0], np.where(face_mask==1)[1]].mean(axis=2)
        return timeseries
    # plot
    images_response_mean = []
    images_response_std = []
    for index,i in enumerate(order_list):
        data = Data[i]
        images_response = stats_value(data)
        images_response_mean.append(images_response.mean(0)[-1])
        images_response_std.append(images_response.std(0)[-1]/np.sqrt(images_response.shape[0]))
    plt.figure(figsize=(4,4), dpi=100)
    plt.bar(range(10), images_response_mean, color=color_list)
    plt.errorbar(range(10), images_response_mean, 
                yerr=images_response_std,
                fmt='.', ecolor='black',elinewidth=2,capsize=4)
    plt.xticks(range(10),legend_list, rotation=70, fontsize=10)
    plt.ylabel('activation', fontsize=10)
                                   
def plot_ge_stable_state_bar(Data, order_list, legend_list, color_list):
    def stats_value(x):
        timeseries = x[:,:,np.where(face_mask==1)[0], np.where(face_mask==1)[1]].mean(axis=2)
        return timeseries
    # plot
    images_response_mean = []
    images_response_std = []
    for index,i in enumerate(order_list):
        data = Data[i]
        images_response = stats_value(data)
        images_response_mean.append(images_response.mean(0)[-1])
        images_response_std.append(images_response.std(0)[-1]/np.sqrt(images_response.shape[0]))
    plt.figure(figsize=(4,4), dpi=100)
    plt.bar(range(8), images_response_mean, color=color_list)
    plt.errorbar(range(8), images_response_mean, 
                yerr=images_response_std,
                fmt='.', ecolor='black',elinewidth=2,capsize=4)
    plt.xticks(range(8),legend_list, rotation=70, fontsize=10)
    plt.ylabel('activation', fontsize=10)                                
        
    
lamda = 0.023
dir_name = './lamda_'+str(lamda)+'/'
images_response_path_list = [dir_name+'se_face_feedback_Dynamic_states.npy',
                             dir_name+'ge_object_feedback_Dynamic_states.npy',
                             dir_name+'f4n_object_feedback_Dynamic_states.npy']

### plot timeseris
Dynamic_states = np.load(images_response_path_list[0])
Data = []
for i in np.arange(0,200,20):
    Data.append(Dynamic_states[i:i+20])
plot_condition_timeseries(Data,
                          order_list=[4,2,3,5,0,9,1,6,7,8],
                          legend_list=['face','cat','dog','lemon','ambulance','store','backpack','pitcher','plane','speaker'],
                          color_list=['red','darkred','indianred','orange','grey','deepskyblue','royalblue','blue','darkblue','black'],
                          label_colation=(0.6,0.7))


Dynamic_states = np.load(images_response_path_list[1])
Data = []
Data.append(Dynamic_states[:10])
Data.append(Dynamic_states[10:20])
Data.append(Dynamic_states[20:30])
Data.append(Dynamic_states[55:65])
Data.append(Dynamic_states[35:45])
Dynamic_states = np.load(images_response_path_list[2])
for i in np.arange(0,75,15):
    Data.append(Dynamic_states[i:i+15])
plot_condition_timeseries(Data,
                          order_list=[0,1,3,5,6,7,8,9],
                          legend_list=['cat face','dog face','tiger face','front','profile','cheek','back','tool'],
                          color_list=['bisque','gold','yellow','red','darkred','indianred','orange','darkblue'])



  
