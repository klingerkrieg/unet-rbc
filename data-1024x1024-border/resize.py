import cv2

for i in range(1,31):
    name = 'blood_smear_%d.png' %i
    print(name)
    try:
        image = cv2.imread(name)
        res = cv2.resize(image, (1024,1024) ,interpolation = cv2.INTER_AREA)
        cv2.imwrite('blood_smear_%d.png' % i, res)
    except:
        print("Erro")