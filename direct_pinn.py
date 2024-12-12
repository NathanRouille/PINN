import torch
import torch.autograd as autograd
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import cm
from pyDOE import lhs

#mettre le dtype par défaut en float32
torch.set_default_dtype(torch.float)

#création d'une graine d'aléatoire pour assurer la reproductibilité de l'entraînement
torch.manual_seed(1234)
np.random.seed(1234)

#utilisation privilégiée du gpu sur le cpu pour un temps de calcul plus faible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device == 'cuda':
    print(torch.cuda.get_device_name())


class NN(nn.Module): #classe de l'objet réseau de neurones

    def __init__(self, layers):

        # appelle __init__ de la classe parent nn.Module
        super().__init__()

        # utilisation d'une fonction d'activation tangente hyperbolique
        self.activation = nn.Tanh()

        # fonction perte erreur quadratique moyenne
        self.loss_function = nn.MSELoss(reduction='mean')

        # initialise le réseau de neurones comme une liste
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        # compteur d'itération lors de l'entraînement
        self.iter = 0

        for i in range(len(layers) - 1):
            # xavier normale initialisation des poids : distribution normale de moyenne 0 et d'ecart type sigma = gain*sqrt(2/(entrées + sorties))
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)

            # biais initialisés à 0
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x): #méthode de calcul des sorties à partir d'entrées

        # transfomartion du vecteur d'entrée en tenseur
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)

        # envoie des torseurs des valeurs min/max des entrées au cpu/gpu
        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)

        # scaling de l'entrée sur le segment [0,1]
        x = (x - l_b) / (u_b - l_b)

        # convertir le torseur d'entrée en float
        a = x.float()

        for i in range(len(layers) - 2):
            # calcul des neurones de la couche i par rapport à la précedente
            z = self.linears[i](a)

            # application de la fonction d'activation aux neurones
            a = self.activation(z)

        a = self.linears[-1](a)

        return a

    def loss_data(self, x, y): #fonction perte de la distance entre la sortie et la valeur attendue

        loss_data = self.loss_function(self.forward(x), y)
        return loss_data

    def loss_PDE(self, input_collocation): #fonction perte de la distance entre la sortie et les équation à dérivées partielles

        # points auxquels on évalue les equations à derivee partielles
        input_col = input_collocation.clone()

        input_col.requires_grad = True

        #calcul de la sortie des entrées
        output_col = self.forward(input_col)

        #dénormalisation de T et V pour assurer le sens physique des grandeurs
        t_col,v_col = output_col[:, [0]]*ecart_temp + moy_temp,output_col[:, [1]]*ecart_vit + moy_vit

        #calcul des dérivées premieres et secondes de la température par rapport à x/y/z avec l'algorithme de backpropagation
        t_x_y_z = autograd.grad(t_col, input_col, torch.ones([input_collocation.shape[0],1]).to(device), retain_graph=True, create_graph=True)[0]
        t_xx_yy_zz = autograd.grad(t_x_y_z, input_col, torch.ones(input_collocation.shape).to(device), create_graph=True)[0]

        # calcul des dérivées premieres et secondes de la vitesse par rapport à x/y/z avec l'algorithme de backpropagation
        v_x_y_z = autograd.grad(v_col, input_col, torch.ones([input_collocation.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
        v_xx_yy_zz = autograd.grad(v_x_y_z, input_col, torch.ones(input_collocation.shape).to(device), create_graph=True)[0]

        t = output_col[:, [0]]*ecart_temp + moy_temp
        t_z = t_x_y_z[:, [2]]
        t_xx = t_xx_yy_zz[:, [0]]
        t_yy = t_xx_yy_zz[:, [1]]
        t_zz = t_xx_yy_zz[:, [2]]

        v = output_col[:, [1]]*ecart_vit + moy_vit
        v_z = v_x_y_z[:, [2]]
        v_xx = v_xx_yy_zz[:, [0]]
        v_yy = v_xx_yy_zz[:, [1]]

        #equations de conservation de la masse, navier stokes et premier principe de la thermodynamique en régime permanent et dans l'approximation de boussinesq
        mass = v_z
        momentum = eta * (v_xx + v_yy) + rho*g*(t-t0)/t0
        energy = rho*cp*v*t_z - lambda_c * (t_xx + t_yy + t_zz)

        #calcul des distances quadratiques moyennes des equations à 0
        loss_mass = self.loss_function(mass, zeros)
        loss_momentum = self.loss_function(momentum, zeros)
        loss_energy = self.loss_function(energy, zeros)

        return loss_mass,loss_momentum,loss_energy


    def loss(self, x, y, input_collocation): #fonction perte totale
        loss_data = self.loss_data(x, y)
        loss_mass,loss_momentum,loss_energy = self.loss_PDE(input_collocation)

        loss_tot = a*loss_data + b*loss_mass + c*loss_momentum + d*loss_energy

        return loss_tot


    def closure(self): #fonction pas de l'optimiseur

        optimizer.zero_grad()

        # calcul de la fonction perte
        loss = self.loss(input_training, output_training, input_collocation)

        # algorithme de backpropagation pour modifier les paramètres en fonction du gradient de la fonction perte
        loss.backward()

        # incrémentation du compteur d'itérations
        self.iter += 1

        # toutes les 100 itérations on affiche la valeur de la fonction perte et les écarts à la réalité
        if self.iter % 100 == 0:
            error_temp,error_vit, _ = PINN.test()

            print(loss.item(), error_temp.item(),error_vit.item())

        return loss


    def test(self): #méthode de test des écarts entre prédiction et réalité avec des donées de test après l'entraînement

        # calcul de la sortie à partir des entrées test
        output_pred = self.forward(input_test)

        #calcul de l'erreur relative de température et vitesse avec la norme 2
        error_temp,error_vit = torch.linalg.norm((output_test[:,0:1] - output_pred[:,0:1]), 2) / torch.linalg.norm(output_test[:,0:1], 2) , torch.linalg.norm((output_test[:,1:2] - output_pred[:,1:2]), 2) / torch.linalg.norm(output_test[:,1:2], 2)

        output_pred = output_pred.cpu().detach().numpy()

        return error_temp,error_vit, output_pred


#parametres adaptables (hyperparametres)
a, b, c, d = 1, 1, 1, 1 #coefficients des différentes fonctions pertes
layers = np.array([3,8,8,8,8,8,8,2]) #couches et nombres de neurones : 6x8 couches cachées
steps=10000 #nombre maximal d'itérations de l'optimiseur
lr=0.01 #learning rate = vitesse de modification des paramètres : w(i+1) = w(i) - lr* ∂(fonction_perte)/∂w(i)

#constantes
g = 9.81 #constante de gravitation terrestre
rho = 1.25 #masse volumique de l'air à température ambiante
cp = 1006 #capacité thermique massique à pression constante de l'air
eta = 18e-6 #viscosité cinématique de l'air
lambda_c = 25e-3 #conductivité thermique de l'air
t0 = 293.15 #température extérieure

#domaine de modélisation
lb = np.array([0,0,0])  #valeurs minimales de x,y,z
ub =  np.array([0.5,0.06,1]) #valeurs maximales de x,y,z


inp = []
vit = []
temp = []

#extraction des données de vitesse et de température
with open('comsol_simulation_speed.txt') as W:
    lines = W.readlines()
    for line in lines:
        L = line.split()
        if L[3] != 'NaN':
            L = list((map(float,L)))
            if L[1]>0:
                inp.append(L[:3])
                vit.append([L[3]])

with open('comsol_simulation_temperature.txt') as T:
    lines = T.readlines()
    for line in lines:
        L = line.split()
        L = list((map(float,L)))
        if L[1]>0:
            temp.append([L[3]])

input = np.array(inp)
vitesse = np.array(vit)
temperature = np.array(temp)
output = np.concatenate((temperature,vitesse),axis=1)

#training points
TP = 100 #nombre de points parmis les mesures
idx = np.random.choice(input.shape[0], TP, replace=False) #indices random parmis tous les indices des lignes des data
input_training = input[idx, :] #input correspondants aux lignes choisies
output_training = output[idx,:] #output correspondants aux lignes choisies

#collocation points
CP = 10000 #nombre de collocation points
input_collocation = lb + (ub-lb)*lhs(3,CP) #Latin Hypercube sampling méthode statistique génération aléatoires en plusieurs dimensions
input_collocation = np.vstack((input_collocation, input_training)) #on ajoute les training points aux collocation points

#test data
TD = 2000 #nombre de points parmis les mesures
idx_test = np.random.choice(input.shape[0], TD, replace=False) #indices random parmis tous les indices des lignes des data
input_test_np = input[idx_test, :] #input correspondants aux lignes choisies
output_test_np = output[idx_test,:] #output correspondants aux lignes choisies

#normaliser les données
moy_temp = np.mean(output_training[:,0])
moy_vit = np.mean(output_training[:,1])
ecart_temp = np.std(output_training[:,0])
ecart_vit = np.std(output_training[:,1])
output_training = np.c_[(output_training[:,0] - moy_temp)/ecart_temp,(output_training[:,1] - moy_vit)/ecart_vit]
output_test_np = np.c_[(output_test_np[:,0] - moy_temp)/ecart_temp,(output_test_np[:,1] - moy_vit)/ecart_vit]
temperature_test = output_test_np[:,0]
vitesse_test = output_test_np[:,1]

#convertir arrays en tenseurs et envoyer au CPU/GPU
input_collocation = torch.from_numpy(input_collocation).float().to(device)
input_training = torch.from_numpy(input_training).float().to(device)
output_training = torch.from_numpy(output_training).float().to(device)
input_test = torch.from_numpy(input_test_np).float().to(device)
output_test = torch.from_numpy(output_test_np).float().to(device)

#torseur avec autant de 0 que de points de collocation pour faire la distance des PDE à 0
zeros = torch.zeros(input_collocation.shape[0], 1).to(device)


PINN = NN(layers) #creation PINN un objet reseau de neurones de taille layers

PINN.to(device) #envoie de PINN au gpu/cpu

print(PINN) #résumé des caractéristiques du réseau de neurones

params = list(PINN.parameters()) #paramètres du réseau à modifier : poids et biais

#Optimisiation avec le L-BFGS Optimizer
optimizer = torch.optim.LBFGS(PINN.parameters(), lr,max_iter=steps,max_eval=None,tolerance_grad=1e-11,tolerance_change=1e-11,history_size=100,line_search_fn='strong_wolfe')

debut = time.time()

optimizer.step(PINN.closure) #utilisation de la méthode closure du réseau de neurones par l'optimiseur

fin = time.time()
print('Temps entraînement: %.2f' % (fin - debut))

#précision du modèle
error_temp,error_vit, output_prediction = PINN.test()

print('Erreur test température: %.5f' % (error_temp))
print('Erreur test vitesse: %.5f' % (error_vit))


def plot3D(data,titre): #fonction pour afficher en 3D des données
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((np.ptp(x), np.ptp(z), 0.3))
    surf3 = ax.plot_trisurf(x, z, data, cmap=cm.jet, linewidth=0)
    plt.title(titre)
    fig.colorbar(surf3)
    fig.tight_layout()
    plt.show()
    return

x = input_test_np[:,[0]].flatten()
y = input_test_np[:,[1]].flatten()
z = input_test_np[:,[2]].flatten()

#dénormalisation des prédictions du réseau de neurones
temperature_pred = output_prediction[:,[0]].flatten()*ecart_temp + moy_temp
vitesse_pred = output_prediction[:,[1]].flatten()*ecart_vit + moy_vit

#affichage des température/vitesse réelles et prédites
plot3D(temperature_test.flatten()*ecart_temp + moy_temp,'Temperature réelle')
plot3D(temperature_pred,'Temperature prédite')
plot3D(vitesse_test.flatten()*ecart_vit + moy_vit,'Vitesse réelle')
plot3D(vitesse_pred,'Vitesse prédite')
