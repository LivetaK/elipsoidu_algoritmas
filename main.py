import numpy as np

def bit_length(x): # si funkcija skaiciuoja, kieik bitu reikia kiekvienam skaiciui uzkoduoti
    return int(np.ceil(np.log2(abs(x) + 1))) if x != 0 else 1 

def h_matrix(n, L): # skaiciuojama pradine h matrica
    return np.eye(n) * (n ** 2) * (2 ** (2 * L))

def constraint_violation(center, epsilon): # tikrinama kurie constraints yra pazeidziama,
    for i, (a, b) in enumerate(constraints):#  randamas pirmas pazeistas ir grazinamas jo indeksas
        if float((a @ center)[0]) > b + epsilon:
            return i
    return None

def update(center, h, a, n): # sita funkcija updatina centra ir h matrica, viskas pagal formules

    Ha = h @ a
    aHa = float((a.T @ Ha).item())
    if aHa < 0: #kad nebutu traukiama saknis is neigiamo skaiciaus
        return None, None
    denominator = np.sqrt(aHa)
    if denominator < 1e-6: # kad nebutu dalybos is nulio, sitas reiskia, kad elispe yra plokscia ir nebera kur eiti daugiau
        return None, None
    new_center = (center - (1 / (n + 1)) * Ha / denominator)

    outer = Ha @ Ha.T
    new_h = (n**2 / (n**2 - 1)) * (h - (2 / (n + 1)) * outer / aHa)

    return new_center, new_h


raw_constraints = [ # pirmieji constraintai, a vektorius - koefizientai, b - reiksmes
    (np.array([-1, 1, -1, -1]), 8),
    (np.array([2, 4, 0, 0]), 10),
    (np.array([0, 0, 1, 1]), 3),
    (np.array([-1, 0, 0, 0]), 0),
    (np.array([0, -1, 0, 0]), 0),
    (np.array([0, 0, -1, 0]), 0),
    (np.array([0, 0, 0, -1]), 0)
]

L = max(bit_length(x) for a, b in raw_constraints for x in a.tolist() + [b]) #pagal formule skaiciuojam L reiksme

epsilon = 2 ** (-2 * L)
constraints = [(a, b + epsilon) for a, b in raw_constraints] #pridedam epsilon prie b kad biski susvelnintumem apribojimus
n = len(constraints[0][0]) #skaiciuojam kiek yra x, kad zinotumem kurios eiles matrica
K = 16 * n * (n + 1) * L # pagal formule, naudosim max iteraciju skaiciui
center = np.zeros((n, 1)) #pirmas centras - koordinaciu plokstumos pradziios taskas susidedaantis is 0
h = h_matrix(n, L) # apskaiciuojam PIRMA matrica

c = np.array([2, -3, 0, -5]).reshape(-1, 1) #apsibreziam c vektoriu, kuris yra musu tikslo funkcijos koeficientu vektorius

best_x = None
best_value = float('inf')

itt = 1

for k in range(K):

    i = constraint_violation(center, epsilon) # grazina kuris apribojimas buvo pazeistas, jeigu nei vienas, grazina None
    
    if i is None: # jeigu apribojimai nebuvo pazeisti
        value = float((c.T @ center).item()) #skaiciuojam kokia tikslo funkcijos reiksme tame taske
        if value < best_value: # jei ji geresne, negu geriasusia
            if abs(value - best_value) < epsilon: #tikrinam ar pakankamas pokytis
                break #jei ne - breakinam
            best_value = value # jei pokytis yra pakankamas, tai tampa musu naujausia geriausia reiksme
            best_x = center.copy()
        
        a = c.copy() # kaip a vektoriu pasiimam c vektoriu, nes bandysim traukti elipse geresnio takso link
    else:
        a = constraints[i][0].reshape(-1, 1) # jeigu apribojimai buvo pazeisti, tada pasiimam a reiksmes to vektoriaus, kurio apribojimai buvo pazeisti

    center, h = update(center, h, a, n) # skaiciuojam nauja centra ir matrica link geresnes krypties
    if center is None or h is None: # jeigu buvo pazeistos kazkokios taisykles is upodate funkcijos, aka bandoma traukti sakni is neigiamo
        break # skaiciaus arba dalyba is nulio, vadinasi breakinam
    itt += 1



print("Optimalus x:")
for i in range(len(best_x)):
    print(f'x{i} = {best_x[i]}')
print()
print("Maziausia reiksme:", best_value)
print()
print('Iteraciju skaicius - ', itt)
