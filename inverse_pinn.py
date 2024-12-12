import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
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

    def loss_equ(self, input_collocation): #fonction perte de la distance entre la sortie et le modèle mathématique

        input_col = input_collocation.clone()

        # calcul de la sortie des entrées
        output_col = self.forward(input_col)

        # dénormalisation de la température pour assurer le sens physique des grandeurs
        output_col = output_col*ecart + moy

        #différence entre la réalité et la prédiction
        temp = output_col - input_col[:,[0]]*(1 - torch.exp(-input_col[:,[2]]*h1.item()*h2.item()/(rho*v0.item()*cp*e*(h1.item() + h2.item()))))/h2.item() - input_col[:,[1]]

        #distance quadratique moyenne de la différence à 0
        loss_temp = self.loss_function(temp, zeros)

        return loss_temp

    def loss(self, x, y, input_collocation): #fonction perte totale
        loss_data = self.loss_data(x, y)
        loss_temp = self.loss_equ(input_collocation)

        loss_tot = a*loss_data + b*loss_temp

        return loss_tot

    def closure(self): #fonction pas de l'optimiseur

        optimizer.zero_grad()

        # calcul de la fonction perte
        loss = self.loss(input_training, output_training, input_collocation)

        # algorithme de backpropagation pour modifier les paramètres en fonction du gradient de la fonction perte
        loss.backward()

        #incrémentation du compteur d'itérations
        self.iter += 1

        # toutes les 10 itérations on affiche la valeur de la fonction perte et l'écart à la réalité et la valeur des paramètres du problème inverse
        if self.iter % 10 == 0:
            error_temp, _ = PINN.test()

            print(loss.item(), error_temp.item(),h1.item(),h2.item(),v0.item())

        return loss

    def test(self): #méthode de test des écarts entre prédiction et réalité avec des donées de test après l'entraînement

        #calcul de la sortie à partir des entrées test
        output_pred = self.forward(input_test)

        # calcul de l'erreur relative de température avec la norme 2
        error_temp = torch.linalg.norm((output_test - output_pred), 2) / torch.linalg.norm(output_test, 2)
        output_pred = output_pred.cpu().detach().numpy()

        return error_temp, output_pred


#parametres adaptables (hyperparametres)
a, b= 1, 1#coefficients des différentes fonctions pertes
layers = np.array([3,10,10,10,10,10,1]) #couches et nombres de neurones
steps=10000 #nombre maximal d'itérations de l'optimiseur
lr=0.01 #learning rate = vitesse de modification des paramètres : w(i+1) = w(i) - lr* ∂(fonction_perte)/∂w(i)

#constantes
e = 0.06 #épaisseur de la couche d'air
rho = 1.25 #masse volumique de l'air à température ambiante
cp = 1006 #capacité thermique massique à pression constante de l'air

#initialisation des coefficients à déterminer par problème inversé
h1 = 10.0 #coefficient conducto-convectif entre l'air intérieur et la plaque d'aluminium
h2 = 16.0 #coefficient conducto-convectif entre l'air extérieur et le bois
v0 = 0.5 #vitesse de l'air sur la ligne de champs (constante d'après la loi de conservation de la masse dans l'approximation de boussinesq)

#domaine de modélisation
lb = np.array([550,5+273.15,0])  # valeurs minimales de Ps,T0,z
ub =  np.array([750,25+273.15,1]) # valeurs maximales Ps,T0,z

inp = []
vit = []
temp = []

#extraction des données de température
with open('real_system_mesures') as M:
    lines = M.readlines()
    for line in lines:
        L = line.split()
        L = list((map(float,L)))
        inp.append([L[0],L[1]+273.15,L[2]/100])
        temp.append([L[3]+273.15])

input = np.array(inp)
output = np.array(temp)

#training points
TP = 30 #nombre de points parmis les mesures
idx = np.random.choice(input.shape[0], TP, replace=False) #indices random parmis tous les indices des lignes des data
input_training = input[idx, :] #input correspondants aux lignes choisies
output_training = output[idx,:] #output correspondants aux lignes choisies

#collocation points
CP = 10000 #nombre de collocation points
input_collocation = lb + (ub-lb)*lhs(3,CP) #Latin Hypercube sampling méthode statistique génération aléatoires en plusieurs dimensions
input_collocation = np.vstack((input_collocation, input_training)) #on ajoute les training points aux collocation points

#test data
TD = 20 #nombre de points parmis les mesures
idx_test = np.random.choice(input.shape[0], TD, replace=False) #indices random parmis tous les indices des lignes des data
input_test_np = input[idx_test, :] #input correspondants aux lignes choisies
output_test_np = output[idx_test,:] #output correspondants aux lignes choisies


#normaliser les données
moy = np.mean(output_training)
ecart = np.std(output_training)
output_training = (output_training - moy)/ecart
output_test_np = (output_test_np - moy)/ecart


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

#ajouter les coefficients à déterminer aux paramètres du réseau de neurones
h1 = nn.Parameter(torch.tensor([h1], requires_grad=True).float().to(device))
h2 = nn.Parameter(torch.tensor([h2], requires_grad=True).float().to(device))
v0 = nn.Parameter(torch.tensor([v0], requires_grad=True).float().to(device))
PINN.register_parameter('h1', h1)
PINN.register_parameter('h2', h2)
PINN.register_parameter('v0', v0)


print(PINN) #résumé des caractéristiques du réseau de neurones

params = list(PINN.parameters()) #paramètres du réseau à modifier : poids,biais et coefficient variables (h1,h2,v0)

#Optimisiation avec le L-BFGS Optimizer
optimizer = torch.optim.LBFGS(PINN.parameters(), lr,max_iter=steps,max_eval=None,tolerance_grad=1e-11,tolerance_change=1e-11,history_size=100,line_search_fn='strong_wolfe')

debut = time.time()

optimizer.step(PINN.closure) #utilisation de la méthode closure du réseau de neurones par l'optimiseur

fin = time.time()
print('Temps entraînement: %.2f' % (fin - debut))

#précision du modèle
error_temp, output_prediction = PINN.test()

print('Erreur test température: %.5f' % error_temp)

Ps = input_test_np[:,0].flatten()
t0 = input_test_np[:,1].flatten()
z = input_test_np[:,2].flatten()

#dénormalisation des prédictions du réseau de neurones
temperature_pred = output_prediction.flatten()*ecart + moy


#affichage de la température réelle/prédite par le réseau
plt.scatter(z,temperature_pred,label='prediction')
plt.scatter(z, output_test_np.flatten()*ecart + moy,label='réalité')
plt.xlabel('hauteur(en m)')
plt.ylabel('temperature(en K)')
plt.title('Température en fonction de la hauteur')
plt.legend()
plt.show()

#calcul de la température avec le modèle mathématique simplifié et les valeurs des coefficients adaptés par le réseau de neurones
equation = Ps*(1 - np.exp(-z*h1.item()*h2.item()/(rho*v0.item()*cp*e*(h1.item() + h2.item()))))/h2.item() + t0

#affichage de la température réelle/modèle mathématique avec les paramètres adaptés
plt.scatter(z,equation,label='equation')
plt.scatter(z, output_test_np.flatten()*ecart + moy,label='réalité')
plt.xlabel('hauteur(en m)')
plt.ylabel('temperature(en K)')
plt.title('Température en fonction de la hauteur')
plt.legend()
plt.show()
