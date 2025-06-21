## Conseils de prise de vue

Pour que la calibration fonctionne avec précision, suivez ces règles :

1. **Variez les angles** : prenez des photos de face, en biais, en diagonale.
2. **Changez la position du damier** dans l’image (centre, bords, coins).
3. **Couvrez toute la surface de l’image** : proche et éloigné (plus ou moins).
4. **Lumière homogène** : pas d’ombres ni de reflets.
5. **Images nettes** : pas de flou.
6. **Inclinaison raisonnable** : évitez les angles trop extrêmes.
7. **Damier plat** : évitez les plis ou les courbures.
8. **Minimum 10–15 images** : idéalement jusqu’à 20.
9. **Tous les coins visibles** : le damier doit être entièrement détectable.
10. **Utilisez une bonne résolution** : pour une meilleure précision (.jpg ou .png).

Voir le [pattern OpenCV](https://github.com/opencv/opencv/blob/master/doc/pattern.png).

Pour l'impression du pattern :
1. **Imprimer l'image en taille réelle** : éviter le redimensionnement à l'impression.
2. **Vérifier la taille des carrés du damier** : paramètre --sqr_size à fournir au script calibrate.py.

Vous pouvez utiliser un autre damier, dans ce cas, il faudra également modifier les paramètres --cols et --rows du script calibrate.py.
