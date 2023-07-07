import cv2

for i in range(1,31):
    name = 'blood_smear_%d.png' %i
    print(name)
    try:
        image = cv2.imread(name)
        res = (255-image)
        cv2.imwrite('blood_smear_%d.png' % i, res)
    except:
        print("Erro")