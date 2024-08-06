import numpy as np

import torch

import torch.nn as nn

import matplotlib.pyplot as plt

import os   

import pandas as pd

import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

 #%% Check GPU

print(f"Pytorch Version: {torch.__version__}")

print("GPU is", "available" if torch.cuda.is_available() else "NOT AVAILABLE")

 

# print(torch.cuda.get_device_name(torch.cuda.current_device()))  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

#%%

class Net(nn.Module):

 

    def __init__(self, n_input, n_output, n_layer, n_nodes):

        super(Net, self).__init__()

        self.n_layer = n_layer

       

        self.Input = nn.Linear(n_input, n_nodes)   # linear layer

        nn.init.xavier_uniform_(self.Input.weight) # wigths and bias initiation

        nn.init.normal_(self.Input.bias)

       

        self.Output = nn.Linear(n_nodes, n_output)

        nn.init.xavier_uniform_(self.Output.weight)

        nn.init.normal_(self.Output.bias)

 

        self.Hidden = nn.ModuleList() # hidden layer list

        for i in range(n_layer):

            self.Hidden.append(nn.Linear(n_nodes, n_nodes))

        for layer in self.Hidden:

            nn.init.xavier_uniform_(layer.weight)

            nn.init.normal_(layer.bias)

 

    def forward(self, x):

        y = torch.tanh(self.Input(x)) # tanh activation function

        for layer in self.Hidden:

            y = torch.tanh(layer(y))

        y = self.Output(y)

        return y



#%%
 
 

def derivative(x, Net,func,order):

   

    w = Net(x)*func(x).view(-1,1) # displacement

   

    if order == '0':

        return w

   

    else:

        dw_xy = torch.autograd.grad(w, x, torch.ones_like(w),

                                    retain_graph=True, create_graph=True)

        dw_x = dw_xy[0][:,0].view(-1,1)

        dw_y = dw_xy[0][:,1].view(-1,1)

       

        if order == '1':

            return w, dw_x, dw_y

   

        else:

            dw_xxy = torch.autograd.grad(dw_x, x, torch.ones_like(dw_x),

                                         retain_graph=True, create_graph=True)

            dw_xx = dw_xxy[0][:,0].view(-1,1)

            dw_xy = dw_xxy[0][:,1].view(-1,1)

            dw_yy = torch.autograd.grad(dw_y, x, torch.ones_like(dw_y), retain_graph=True,

                                        create_graph=True)[0][:,1].view(-1,1)

            return w, dw_x, dw_y, dw_xx, dw_yy, dw_xy

#%%
 
 

def derivative1(x, Net,func,order):

   

    w = Net(x)*func(x).view(-1,1)+1 # displacement

   

    if order == '0':

        return w

   

    else:

        dw_xy = torch.autograd.grad(w, x, torch.ones_like(w),

                                    retain_graph=True, create_graph=True)

        dw_x = dw_xy[0][:,0].view(-1,1)

        dw_y = dw_xy[0][:,1].view(-1,1)

       

        if order == '1':

            return w, dw_x, dw_y

   

        else:

            dw_xxy = torch.autograd.grad(dw_x, x, torch.ones_like(dw_x),

                                         retain_graph=True, create_graph=True)

            dw_xx = dw_xxy[0][:,0].view(-1,1)

            dw_xy = dw_xxy[0][:,1].view(-1,1)

            dw_yy = torch.autograd.grad(dw_y, x, torch.ones_like(dw_y), retain_graph=True,

                                        create_graph=True)[0][:,1].view(-1,1)

            return w, dw_x, dw_y, dw_xx, dw_yy, dw_xy
#%%

# Domain data

# x, y = np.meshgrid(np.linspace(0.05,1.95,80), np.linspace(0.05,0.95,40))
# df = pd.read_csv('ufielddata.csv').to_numpy()
# temp = df[:,:2]


# data = torch.tensor(temp, dtype=torch.float32, requires_grad=True,device=device)

y1 = torch.linspace(0, 0.01, 11)
y2 = torch.linspace(0.01, 0.5, 31)[1:]
y3 = torch.linspace(0.5, 0.99, 31)[1:]
y4 = torch.linspace(0.99, 1, 11)
y = torch.cat((y1, y2, y3, y4), dim=0)

x, y = torch.meshgrid(torch.linspace(0, 2, 101), y)


# df = pd.read_csv('ufielddata.csv')
# temp = df.to_numpy()


# x = temp[:, 0]
# y = temp[:, 1]

# data = torch.tensor([x, y], dtype=torch.float32, requires_grad=True, device=device)

# x, y = np.meshgrid(np.linspace(0,2,201), np.linspace(0,1,101))
#%%%

x = x.reshape(-1,1) #- x.mean()

y = y.reshape(-1,1) #- y.mean()

data = np.hstack([x, y])
min_val = np.min(data[:, 1])
max_val = np.max(data[:, 1])
min1_val = np.min(data[:, 0])
max1_val = np.max(data[:, 0])  
    
# find boundary

idx_b = np.where((data[:,1]==min(data[:,1])))
idx_b1 = np.where((data[:,1]==min(data[:,1]))& (data[:, 0] >=0.1) & (data[:, 0]<=1.9))
idx_b2 = np.where((data[:,1]==min(data[:,1]))& (data[:, 0] <0.1))
idx_b3 = np.where((data[:,1]==min(data[:,1])) & (data[:, 0]>1.9))

idx_u = np.where((data[:,1]==max(data[:,1])))
idx_u1 = np.where((data[:,1]==max(data[:,1]))& (data[:, 0] >=0.1) & (data[:, 0]<=1.9))
idx_u2 = np.where((data[:,1]==max(data[:,1]))& (data[:, 0] <0.1))
idx_u3 = np.where((data[:,1]==max(data[:,1]))& (data[:, 0]>1.9))

idx_l = np.where((data[:,0]==min(data[:,0])))                
idx_l1 = np.where((data[:,0]==min(data[:,0])) & (data[:, 1] != 0) & (data[:, 1] != 1))

idx_r = np.where((data[:,0]==max(data[:,0]))) 
idx_r1 = np.where((data[:,0]==max(data[:,0])) & (data[:, 1] != 0) & (data[:, 1] != 1))

data = torch.tensor(data, dtype=torch.float32, requires_grad=True, device=device)



#%%

# Construct neural network
Net_psi = Net(2, 1, 9, 20).to(device) # NN for psi displacement
Net_p = Net(2, 1, 9, 20).to(device) # NN for p displacement
Net_c1 = Net(2, 1, 5, 20).to(device) # NN for c displacement
Net_phi = Net(2, 1, 9, 20).to(device) # NN for c displacement
Net_c2 = Net(2, 1, 5, 20).to(device) # NN for c displacement

func_phi = lambda data: torch.abs(data[:,0]-2) # V=0 BC
func_p = lambda data: torch.abs(data[:,0]-1)-1
func_c = lambda data: torch.abs(data[:,0])
func_u = lambda data: torch.abs(data[:,1] - 0.5)-0.5

func_v = lambda data: torch.abs(data[:,1] - 0.5)-0.5
# plain strain parameters
func = lambda data: torch.ones(data[:,0].shape, device=device)

D1=1.97
D2=1


z1=1
z2=-1
sigma=-1
Cc=2494.3
R=1e-2
iteration = 0


loss_hist = []

#%% Model train

epochs = 10000

learning_rate = 1e-4 

adam_optimizer = torch.optim.Adam(list(Net_psi.parameters())+list(Net_p.parameters())+list(Net_c1.parameters())+list(Net_c2.parameters())+list(Net_phi.parameters()), lr=learning_rate)
#lbfgs_optimizer = torch.optim.LBFGS(list(Net_u.parameters())+list(Net_v.parameters())+list(Net_p.parameters()), lr=learning_rate)
 

# lbfgs_optimizer = torch.optim.LBFGS(list(Net_psi.parameters())+list(Net_p.parameters())+list(Net_c1.parameters())+list(Net_c2.parameters())+list(Net_phi.parameters()), lr=learning_rate,

#                               max_iter = 1000,

#                               max_eval = None,

#                               tolerance_grad=1e-11,

#                               tolerance_change=1e-11,

#                               history_size=100,

#                               line_search_fn='strong_wolfe')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam_optimizer, 'min', factor=0.95, min_lr=1e-8)#lr down
loss1_hist = []
loss2_hist = []
loss3_hist = []
loss4_hist = []
loss5_hist = []
loss6_hist = []
loss7_hist = []
loss8_hist = []
loss9_hist = []
loss10_hist = []
loss11_hist = []
loss12_hist = []
loss13_hist = []
loss14_hist = []
loss15_hist = []
loss16_hist = []
loss17_hist = []
loss18_hist = []
loss19_hist = []
loss20_hist = []
loss21_hist = []
loss22_hist = []
loss24_hist = []
loss25_hist = []
time1 = time.time()

for epoch in range(epochs):

    def closure():

        global iteration

        adam_optimizer.zero_grad()
        # lbfgs_optimizer.zero_grad()

       

        # Calculate derivatives

        c1, c1x, c1y, c1xx, c1yy, c1xy = derivative1(data, Net_c1, func_c, '2')
        c2, c2x, c2y, c2xx, c2yy, c2xy = derivative1(data, Net_c2, func_c, '2')
        psi, psi_x, psi_y, psi_xx, psi_yy, psi_xy = derivative(data, Net_psi, func, '2')
        phi, phix, phiy, phixx, phiyy, phixy = derivative(data, Net_phi, func_phi, '2')
        p, px, py, pxx, pyy, pxy = derivative(data, Net_p, func_p, '2')
        
        u = psi_y
        v = -psi_x
        
        u_xy = torch.autograd.grad(u, data, torch.ones_like(u), retain_graph=True, create_graph=True) # diffentiate u with respect to x and y
        v_xy = torch.autograd.grad(v, data, torch.ones_like(v), retain_graph=True, create_graph=True)
        
        ux = u_xy[0][:,0].view(-1,1)#get du/dx
        uy = u_xy[0][:,1].view(-1,1)#get dy/dy

        vx = v_xy[0][:,0].view(-1,1)
        vy = v_xy[0][:,1].view(-1,1)
        
        ux_xy = torch.autograd.grad(ux, data, torch.ones_like(u), retain_graph=True, create_graph=True) # diffentiate u with respect to x and y
        vx_xy = torch.autograd.grad(vx, data, torch.ones_like(v), retain_graph=True, create_graph=True)
        uy_xy = torch.autograd.grad(uy, data, torch.ones_like(u), retain_graph=True, create_graph=True) # diffentiate u with respect to x and y
        vy_xy = torch.autograd.grad(vy, data, torch.ones_like(v), retain_graph=True, create_graph=True)
        
        uxx = ux_xy[0][:,0].view(-1,1)
        # uxy = ux_xy[0][:,1].view(-1,1)
        uyy = uy_xy[0][:,1].view(-1,1)

        vxx = vx_xy[0][:,0].view(-1,1)
        # vxy = vx_xy[0][:,1].view(-1,1)
        vyy = vy_xy[0][:,1].view(-1,1)
        
        L1 = (uxx+uyy-px)-Cc*(c1-c2)*phix #-Sc*(u*ux+v*uy)+
        L2 = (vyy+vxx-py)-Cc*(c1-c2)*phiy #-Sc*(u*vx+v*vy)+
        L3 = D1*c1xx+z1*D1*c1x*phix+z1*D1*c1*phixx-u*c1x+D1*c1yy+z1*D1*c1y*phiy-v*c1y+z1*D1*c1*phiyy
        L4 = D2*c2xx+z2*D2*c2x*phix+z2*D2*c2*phixx-u*c2x+D2*c2yy+z2*D2*c2y*phiy-v*c2y+z2*D2*c2*phiyy
        L5 = (2*R*R)*(phixx+phiyy)+(c1-c2)
        
        # s_rp = -p[idx_r]
        # s_lp = p[idx_l]
        #s_lv = v[idx_l]
        s_ru = -ux[idx_r]
        s_rv = -vx[idx_r]
        s_uu = u[idx_u]
        # s_uu2 = u[idx_u2]
        # s_uu3 = u[idx_u3]
        s_uv = v[idx_u]
        s_bu = u[idx_b]
        # s_bu2 = u[idx_b2]
        # s_bu3 = u[idx_b3]
        s_bv = v[idx_b]
        
        # s_lc1 = c1[idx_l]-1
        s_bc1 = D1*c1y[idx_b]+z1*D1*c1[idx_b]*phiy[idx_b]
        s_uc1 = -D1*c1y[idx_u]-z1*D1*c1[idx_u]*phiy[idx_u]
        s_rc1 = D1*c1x[idx_r]
        
        # s_lc2 = c2[idx_l]-1
        s_bc2 = D2*c2y[idx_b]+z2*D2*c2[idx_b]*phiy[idx_b]
        s_uc2 = -D2*c2y[idx_u]-z2*D2*c2[idx_u]*phiy[idx_u]
        s_rc2 = D2*c2x[idx_r]
        
        s_lphi = phi[idx_l]-1
        s_u1phi = phiy[idx_u1]+1
        s_u2phi = phiy[idx_u2]
        s_u3phi = phiy[idx_u3]
        
        s_b1phi = phiy[idx_b1]-1
        s_b2phi = phiy[idx_u2]
        s_b3phi = phiy[idx_u3]
        
        
        # s_rphi = phi[idx_r]
        
        loss1=torch.mean(L1**2)
        loss2=torch.mean(L2**2)
        loss3=torch.mean(L3**2)
        loss4=torch.mean(L4**2)
        loss5=torch.mean(L5**2)
        
        # loss6=torch.mean(s_rp**2)
        # loss7=torch.mean(s_lp**2)
        loss8=torch.mean(s_ru**2)
        loss9=torch.mean(s_rv**2)
        loss10=torch.mean(s_uu**2)
        loss11=torch.mean(s_uv**2)
        loss12=torch.mean(s_bu**2)
        loss13=torch.mean(s_bv**2)
        
        # loss14=torch.mean(s_lc1**2)
        loss15=torch.mean(s_bc1**2)
        loss16=torch.mean(s_uc1**2)
        loss17=torch.mean(s_rc1**2)
        # loss18=torch.mean(s_lc2**2)
        loss19=torch.mean(s_bc2**2)
        loss20=torch.mean(s_uc2**2)
        loss21=torch.mean(s_rc2**2)
        
        loss22=torch.mean(s_lphi**2)
        loss24=torch.mean(s_u1phi**2)
        loss25=torch.mean(s_b1phi**2)
        # loss26=torch.mean(s_rphi**2)
        loss27=torch.mean(s_u2phi**2)
        loss28=torch.mean(s_u3phi**2)
        loss29=torch.mean(s_b2phi**2)
        loss30=torch.mean(s_b3phi**2)
        
        loss =(loss1+loss2+loss3+loss4+loss5+loss8+loss9+loss10+\
               loss11+loss12+loss13+loss15+loss16+loss17+loss19+loss20+\
                   loss21+loss22+10000*loss24+10000*loss25+loss27+loss28+loss29+loss30)
        # scheduler.step(loss)
    
          
        loss1_hist.append(loss1.item())
        loss2_hist.append(loss1.item())
        loss3_hist.append(loss1.item())
        loss4_hist.append(loss1.item())
        loss5_hist.append(loss1.item())
        loss6_hist.append(loss1.item())
        loss7_hist.append(loss1.item())
        loss8_hist.append(loss1.item())
        loss9_hist.append(loss1.item())
        loss10_hist.append(loss1.item())
        loss11_hist.append(loss1.item())
        loss12_hist.append(loss1.item())
        loss13_hist.append(loss1.item())
        loss14_hist.append(loss1.item())
        loss15_hist.append(loss1.item())
        loss16_hist.append(loss1.item())
        loss17_hist.append(loss1.item())
        loss18_hist.append(loss1.item())
        loss19_hist.append(loss1.item())
        loss20_hist.append(loss1.item())
        loss21_hist.append(loss1.item())
        loss22_hist.append(loss1.item())
        loss24_hist.append(loss1.item())
        loss25_hist.append(loss1.item())

        loss_hist.append(loss.item())

        loss.backward(retain_graph=True)

        iteration += 1

        print('[Epoch: %d] [loss: %.3E] [loss1: %.3E] [loss2: %.3E] [loss3: %.3E] [loss4: %.3E] [loss5: %.3E] [loss6: %.3E] [loss7: %.3E] [loss8: %.3E] [loss9: %.3E] [loss10: %.3E]\
              [loss11: %.3E] [loss12: %.3E][loss13: %.3E] [loss14: %.3E] [loss15: %.3E] [loss16: %.3E] [loss17: %.3E] [loss18: %.3E] [loss19: %.3E] [loss20: %.3E]\
                  [loss21: %.3E] [loss22: %.3E] [loss24: %.3E] [loss25: %.3E]' %\
              (iteration, loss.item(), loss1.item(), loss2.item(), loss3.item(),loss4.item(),loss5.item(),loss5.item(),loss8.item(), loss8.item(), loss9.item(), loss10.item(), \
               loss11.item(), loss12.item(), loss13.item(),loss13.item(),loss15.item(),loss16.item(),loss17.item(), loss13.item(), loss19.item(), loss20.item(), loss21.item(),\
                  loss22.item(),loss24.item(),loss25.item()))
 
        return loss
    # lbfgs_optimizer.step(closure)
   
        # if epoch % 2 == 0:
        #     adam_optimizer.step(closure)
        # else:
        #       lbfgs_optimizer.step(closure)
    
    if epoch > 10000:
        # lbfgs_optimizer.step(closure)
        break
    else:
        adam_optimizer.step(closure)
        scheduler.step(closure())

print('Training Complete')

time2 = time.time()

print(time2-time1)



#%% Predict

psi, psi_x, psi_y, psi_xx, psi_yy, psi_xy = derivative(data, Net_psi, func, '2')
u = psi_y
v = -psi_x
p, px, py, pxx, pyy, pxy = derivative(data, Net_p, func_p, '2')
c1, c1x, c1y, c1xx, c1yy, c1xy = derivative1(data, Net_c1, func_c, '2')
c2, c2x, c2y, c2xx, c2yy, c2xy = derivative1(data, Net_c2, func_c, '2')
phi, phix, phiy, phixx, phiyy, phixy = derivative(data, Net_phi, func_phi, '2')


upred = torch.abs(u.cpu()).detach().numpy()
vpred = v.cpu().detach().numpy()
ppred = p.cpu().detach().numpy()
c1pred = c1.cpu().detach().numpy()
c2pred = c2.cpu().detach().numpy()
phipred = phi.cpu().detach().numpy()

plt.scatter(x, y, c=ppred, cmap='jet')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='u')


#%%
df = pd.read_csv('ufielddata.csv').to_numpy()
temp = df[:,:2]


tdata = torch.tensor(temp, dtype=torch.float32, requires_grad=True,device=device)
psi, psi_x, psi_y, psi_xx, psi_yy, psi_xy = derivative(tdata, Net_psi, func, '2')
p, px, py, pxx, pyy, pxy = derivative(tdata, Net_p, func, '2')
c1, c1x, c1y, c1xx, c1yy, c1xy = derivative1(tdata, Net_c1, func_c, '2')
c2, c2x, c2y, c2xx, c2yy, c2xy = derivative1(tdata, Net_c2, func_c, '2')
phi, phix, phiy, phixx, phiyy, phixy = derivative(tdata, Net_phi, func_phi, '2')
u = psi_y
v = -psi_x
upred = torch.abs(u.cpu()).detach().numpy()
vpred = v.cpu().detach().numpy()
ppred = p.cpu().detach().numpy()
c1pred = c1.cpu().detach().numpy()
c2pred = c2.cpu().detach().numpy()
phipred = phi.cpu().detach().numpy()

 #%%
figure, ax = plt.subplots(3,1)
im=ax[0].scatter(temp[:,0], temp[:,1], c=upred, cmap='jet', vmin=0, vmax=6e-5)
im=ax[1].scatter(temp[:,0], temp[:,1], c=df[:,2], cmap='jet', vmin=0, vmax=6e-5)
im=ax[2].scatter(temp[:,0], temp[:,1], c=df[:,2].reshape(-1,1)-upred, cmap='jet', vmin=0, vmax=np.max(df[:,2]))
# Adding the colorbar
cbaxes = figure.add_axes([0.0, 0.1, 0.03, 0.8]) 
figure.colorbar(im, ax=ax.ravel().tolist(),location='left', cax=cbaxes)
plt.subplots_adjust(hspace=0.5)

#%%
x_values = temp[:, 0]
y_values = temp[:, 1]
x_equals_1_indices = np.where(np.isclose(x_values, 1.0))[0]
  

sample_indices = np.arange(0, len(x_equals_1_indices), 1)

upred_x_equals_1 =upred[x_equals_1_indices][sample_indices]
df_x_equals_1 = df[x_equals_1_indices, 2][sample_indices]

scatter_upred = plt.scatter(y_values[x_equals_1_indices][sample_indices], upred_x_equals_1, c='red', label='upred', marker='s')
# scatter_df = plt.scatter(y_values[x_equals_1_indices][sample_indices], df_x_equals_1, c='blue', label='uFEM', marker='^')



plt.xlabel('y (μm)')
plt.ylabel('u (m/s)')

plt.legend()

plt.show()
#%%
figure, ax = plt.subplots()
im=ax.scatter(temp[:,0], temp[:,1], c=phipred, cmap='jet')#, vmin=0, vmax=1) 

figure.colorbar(im, ax=ax)
#%%
fig,ax = plt.subplots()
plt.yscale("log") 
ax.plot(loss1_hist)
ax.plot(loss2_hist)
ax.plot(loss3_hist)
ax.plot(loss4_hist)
ax.plot(loss5_hist)
ax.plot(loss6_hist)
ax.plot(loss7_hist)
ax.plot(loss8_hist)
ax.plot(loss9_hist)
ax.plot(loss10_hist)
ax.plot(loss11_hist)
ax.plot(loss12_hist)
ax.plot(loss13_hist)
ax.plot(loss14_hist)
ax.plot(loss15_hist)
ax.plot(loss16_hist)
ax.plot(loss17_hist)
ax.plot(loss18_hist)
ax.plot(loss19_hist)
ax.plot(loss20_hist)
ax.plot(loss21_hist)
ax.plot(loss22_hist)
ax.plot(loss24_hist)
ax.plot(loss25_hist)

