# MLOps-ProyectoIndividual
##  Steam Games API 

Esta API proporciona información sobre los juegos de Steam, incluyendo géneros, especificaciones, fechas de lanzamiento, puntuaciones y más. Puedes utilizar esta API para obtener datos relevantes sobre los juegos lanzados en diferentes años.

## Endpoints Disponibles

/ : Devuelve un mensaje de bienvenida.

/genre/{year} : Obtiene los géneros más populares de los juegos lanzados en el año especificado.

/games/{year} : Devuelve la lista de nombres de juegos lanzados en el año especificado.

/specs/{year} : Obtiene las especificaciones más comunes de los juegos lanzados en el año especificado.

/earlyaccess/{year} : Devuelve la cantidad de juegos lanzados con acceso temprano en el año especificado.

/sentiment/{year} : Obtiene el sentimiento predominante en los comentarios de los juegos lanzados en el año especificado.

/metascore/{year} : Devuelve los nombres y puntajes de los juegos con las mejores puntuaciones en el año especificado.

/predict/ : Realiza una predicción del precio de un juego basado en los atributos ingresados.

## Uso de la API

Para utilizar la API, simplemente realiza una solicitud GET a los diferentes endpoints mencionados anteriormente. Puedes proporcionar el año como un parámetro en la URL para obtener información específica de ese año.

**Ejemplos de Uso**

Para obtener los géneros más populares de los juegos lanzados en el año 2021: GET /genre/2021

Para obtener la lista de nombres de juegos lanzados en el año 2022: GET /games/2022

Para obtener las especificaciones más comunes de los juegos lanzados en el año 2023: GET /specs/2023

Para obtener la cantidad de juegos con acceso temprano lanzados en el año 2020: GET /earlyaccess/2020

Para realizar una predicción del precio de un juego: GET /predict/?genre=Action&early_access=true&metascore=80&year=2022

## Instalación y Ejecución en LocalHost.

-Clona el repositorio:

'git clone <https://github.com/Guido097/MLOps-ProyectoIndividual.git>'

-Instala las dependencias con el comando:

'pip install -r requirements.txt'

-Ejecuta la aplicación con el comando:

'uvicorn main:app --reload'


# Notas

-Asegúrate de tener Python 3.x instalado.

-La API utiliza el paquete FastAPI y requiere la instalación de las dependencias especificadas en requirements.txt.

-Algunos endpoints pueden requerir un año válido como parámetro en la URL.


## Ejecución en el deploy en Render.

Para ejecutar la API y poder probar cada una de las funciones debes entrar en la URL <https://mlops-proyectoindividual.onrender.com>.

Este es un pequeño instructivo para el correcto funcionamiento de la API.
1-Para las funciones donde se pide ingresar un año, debe de ingresar uno de los que aparecen en la siguiente lista, ya que son los años que se encuentran dentro de la base de datos, de lo contrario la funcion indicará que hay un error.
"[1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021]"

2-Para la función de predicción es necesario completar con "género", "early_access", "metascore" y "year".

-Para genero se debe ingresar un valor de la siguiente lista:
" ['Action', 'Adventure', 'Animation &amp; Modeling', 'Audio Production', 'Casual', 'Design &amp; Illustration', 'Early Access', 'Education', 'Free to Play', 'Indie', 'Massively Multiplayer', 'Photo Editing', 'RPG', 'Racing', 'Simulation', 'Software Training', 'Sports', 'Strategy', 'Utilities', 'Video Production', 'Web Publishing'] "

-Para early_access se debe ingresar "true" o "false", ya que es un valor booleano.

-Para metascore se debe ingresar un numero de "1" al "100", siguiendo ese formato.

-Para year se repite las indicaciones de las funciones del punto 1. Se debe de ingresar uno de los que aparecen en la siguiente lista:

"[1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021]"


## Video

Enlace del video <https://youtu.be/T5JYU9f8B4Q>


