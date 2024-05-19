import json
from nltk.stem.snowball import SnowballStemmer
import os
import re
import sys
import math
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle

class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [
        ("all", True), ("title", True), ("summary", True), ("section-name", True), ('url', True)
    ]
    def_field = 'all'
    PAR_MARK = '%'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10

    all_atribs = ['urls', 'index', 'sindex', 'ptindex', 'docs', 'weight', 'articles',
                  'tokenizer', 'stemmer', 'show_all', 'use_stemming']

    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes añadir más variables si las necesitas 

        """
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.sindex = {} # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {} # hash para el indice permuterm.
        self.docs = {} # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.weight = {} # hash de terminos para el pesado, ranking de resultados.
        self.articles = {} # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile("\W+") # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer('spanish') # stemmer en castellano
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()
        self.parpos={}

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_snippet(self, v:bool):
        """

        Cambia el modo de mostrar snippet.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_snippet es True se mostrara un snippet de cada noticia, no aplicable a la opcion -C

        """
        self.show_snippet = v


    def set_stemming(self, v:bool):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v



    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):
        """
        Guarda la información del índice en un fichero en formato binario
        
        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """
        Carga la información del índice desde un fichero en formato binario
        
        """
        #info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls


    def index_dir(self, root:str, **args):
        """
        
        Recorre recursivamente el directorio o fichero "root" 
        NECESARIO PARA TODAS LAS VERSIONES
        
        Recorre recursivamente el directorio "root" e indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.multifield = args['multifield']
        self.positional = args['positional']
        self.stemming = args['stem']
        self.permuterm = args['permuterm']

        file_or_dir = Path(root)
        
        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in sorted(files):
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        # anade el documento al self.docs para su uso posterior
                        self.docs[len(self.docs) + 1] = fullname
                        # indexa un documento
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################

        # si se quiere usar stemming, se llama a la función para crear el stemming
        if(self.stemming):
            self.make_stemming()

        # si se quiere usar permuterm, se llama a la función para crear permuterm
        if(self.permuterm):
            self.make_permuterm()
        
        
        
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos 
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article
                
    
    def index_file(self, filename:str):
        """

        Indexa el contenido de un fichero.
        
        input: "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.multifield y self.positional se debe ampliar el indexado

        """
        fields_to_tokenize = []

        if(self.multifield == True):
            for field in self.fields:
                if field[1] and field[0] != 'url':
                    fields_to_tokenize.append(field[0])
        else:
            fields_to_tokenize.append(self.def_field)


        """
            para cada artículo del file
        """
        for i, line in enumerate(open(filename)):
            """
                consigue el artículo separando sus fields
            """
            j = self.parse_article(line)


        #################
        ### COMPLETAR ###
        #################
            """
                inicia la id de los artículos a cero
            """
            artId = 0;
            
            """
                si el artículo todavía no se ha analizado, se añade a los artículos ya vistos
                sumando uno a la longitud de los artículos analizados (artId)
                se guarda en su posición de self.articles su url y título para identificación
                y posterior uso
                también se añade su url a self.urls para podes comprobar que ya se ha analizado
            """
            if(j['url'] not in self.urls):
                artId = len(self.articles) + 1
                self.articles[artId] = [j['url'], j['title']]
                self.urls.add(j['url'])

                """ 
                    si no se quiere usar el índice posicional:
                        1. recorre los fields indicados menos 'url' que no debe tokenizarse
                        2. tokeniza y field y deja el resultado en tk
                        3. si el field no está en el índice, añade el diccionario
                        4. por cada token en tk:
                            4.1 si no está el término en el índice del field, añade la lista y el artId
                            4.2 si está en el índice, añade el artId a la lista del término
                        5. si hace falta se crea el índice para el field url
                        6. si no comprueba si está el url en el índica, si no está crea su lista
                        7. añade el artId a la lista de la url
                """

                # si no se quiere usar el índice posicional 
                if(self.positional == False):
                    # para cada field
                    for field in fields_to_tokenize:
                        tk = self.tokenize(j[field])
                        if(field not in self.index):
                            self.index[field] = {}
                         # para cada token
                        for t in tk:
                           if(t not in self.index[field]):
                                self.index[field][t] = []
                                self.index[field][t].append(artId)
                           else:
                               if(artId not in self.index[field][t]):
                                self.index[field][t].append(artId)   
                    
                    # field url
                    if('url' not in self.index):
                        self.index['url'] = {}
                    if(j['url'] not in self.index['url']):
                        self.index['url'][j['url']] = []
                    self.index['url'][j['url']].append(artId)
                    
                """ 
                    si se quiere usar el índice posicional:
                        1. recorre los fields indicados menos 'url' que no debe tokenizarse
                        2. tokeniza y field y deja el resultado en tk
                        3. si el field no está en el índice, añade el diccionario
                        4. por cada token en tk:
                            4.1 si no está el término en el índice del field, añade su diccionario,
                                añade la lista del artId y la posición correspondiente dentro del artículo.
                            4.2 si está en el índice:
                                4.2.1 si no está el artículo en el término, se añade su lista y la posición
                                4.2.2 si está el artículo en el término, se añade la posición a su lista.
                        5. si hace falta se crea el índice para el field url
                        6. comprueba:
                            6.1 si el término no está en 'url' crea el diccionario
                            6.2 si el término está en 'url':
                                6.2.1 si no está el artículo en el término, crea su lista 
                        7. añade la posición cero a la lista del artículo
                """

                if self.positional:

                    # recorre los fields
                    for field in fields_to_tokenize:
                        tk = self.tokenize(j[field])
                        if field not in self.index:
                            self.index[field] = {}
                        # recorre los tokens consiguiendo sus posiciones
                        for i, t in enumerate(tk):
                            if(t not in self.index[field]):
                                self.index[field][t] = {}
                                self.index[field][t][artId] = []
                                self.index[field][t][artId].append(i)
                            else:
                                if artId not in self.index[field][t]:
                                    self.index[field][t][artId] = []
                                    self.index[field][t][artId].append(i)
                                else:
                                    self.index[field][t][artId].append(i)

                    # field url
                    t = j['url']
                    if 'url' not in self.index:
                        self.index['url'] = {}
                    if t not in self.index['url']:
                        self.index['url'][t] = {}
                    if artId not in self.index['url'][t]:
                        self.index['url'][t][artId] = []
                    self.index['url'][t][artId].append(0)
        


    def set_stemming(self, v:bool):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v


    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()


    def make_stemming(self):
        """
        Crea el indice de stemming (self.sindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE STEMMING.

        "self.stemmer.stem(token) devuelve el stem del token"
        """
        #Por cada field en index. Esto asegura que funcione con multifield
        for field in self.index:
            self.sindex[field] = {}
            for token in self.index[field]:
                #Pasamos cada palabra de index por el stemmer
                stemtoken = self.stemmer.stem(token)
                #Si la palabra no es en el diccionario de stems lo agregamos
                if stemtoken not in self.sindex[field]:
                    #En un principio usaremos sets para que no hayan terminos repetidos
                    self.sindex[field][stemtoken] = set(self.index[field][token])
                else:
                    (self.sindex[field][stemtoken]).update(self.index[field][token])
            #Cuando acabemos con todas las palabras de un field ordenamos cada set y la transformamos en una lista
            for stemtoken in self.sindex[field]:
                self.sindex[field][stemtoken] = list(sorted(self.sindex[field][stemtoken]))
        pass



    
    def make_permuterm(self):
        """

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        """
        #Por cada field en index. Esto asegura que funcione con multifield
        for field in self.index:
            self.ptindex[field] = []
            #Generador de permuterms
            for i in self.index[field]:
                #Añadimos simbolo final de palabra
                cadena = "".join([i,"$"])
                #Añadimos par (permuterm, palabra)
                self.ptindex[field].append((cadena,i))
                for j in range(len(cadena)-1):
                    #Generamos siguiente permuterm
                    cadena = "".join([cadena[-1:],cadena[:-1]])
                    #Añadimos par (permuterm, palabra)
                    self.ptindex[field].append((cadena,i))
            #Ordenamos la lista para facilitar las queries
            self.ptindex[field].sort()
        pass





    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Muestra estadisticas de los indices
        
        """
        #Impresión por pantalla del resultado de la indexación, mostrando primeramente el número de archivos y de artículos.
        print("========================================")
        print(f"Number of indexed files: {len(self.docs)}")
        print("----------------------------------------")
        print(f"Number of indexed articles: {len(self.articles)}")
        print("----------------------------------------")

        #Imprime para los campos seleccionados(en caso de multifield) las estadísticas del índice invertido de términos.
        print("TOKENS:")
        if(self.multifield):
            for field in self.fields:
                if field[1]:
                    print(f'\t# of tokens in "{field[0]}": {len(self.index[field[0]])}')
        else:
            print(f'\t# of tokens in "{self.def_field}":, {len(self.index[self.def_field])}')


        #Imprime para los campos seleccionados(en caso de multifield) las estadísticas del índice invertido de permuterms.
        if(self.permuterm):
            print("----------------------------------------")
            print("PERMUTERMS:")
            if(self.multifield):
                for field in self.fields:
                    if field[1]:
                        print(f'\t# of permuterms in "{field[0]}": {len(self.ptindex[field[0]])}')
            else:
                print(f'\t# of permuterms in "{self.def_field}":, {len(self.ptindex[self.def_field])}')

        #Imprime para los campos seleccionados(en caso de multifield) las estadísticas del índice invertido de stems.
        if(self.stemming):
            print("----------------------------------------")
            print("STEMS:")
            if(self.multifield):
                for field in self.fields:
                    if field[1]:
                        print(f'\t# of stems in "{field[0]}": {len(self.sindex[field[0]])}')
            else:
                print(f'\t# of stems in "{self.def_field}":, {len(self.sindex[self.def_field])}')

        #Imprime si las positionals están o no activadas.
        print("----------------------------------------")
        if(self.positional):
            print("Positional queries are allowed.")
        else:
            print("Positional queries are NOT allowed.")
        
        print("========================================")




    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################
    def hashkey(self,query:str,cont:int):
        key = 0
        for c in query:
            key+=ord(c)

        while(key in self.parpos):
            key+=cont

        return f"{key}"

    def solve_parpos(self,ini,cont,query):
        if(query[cont]=='('):
            ini = cont
        if('(' in query[cont:]):
            cont+=1
            query=self.solve_parpos(ini,cont,query)

        if(ini!=None):
            if(')' in query[cont:] and query[cont-1]=='('):
                while(query[cont]!=')'):
                    cont+=1
                
                key = self.hashkey(query[ini+1:cont],cont)
                self.parpos[key]=self.solve_query(query[ini+1:cont])


                return query[:ini]+key+query[cont+1:]
            else:
                return query
        else:
            return query

    def calculateposting(self,term:str):
        if(':' in term):
                field,name=term.split(':')
                postinglist = self.get_posting(name,field)
        else:
            if(term in self.parpos):
                postinglist=self.parpos[term]
            else:
                postinglist = self.get_posting(term)
        
        if(isinstance(postinglist,dict)):
            return list(postinglist.keys())
        return postinglist


    def solve_query(self, query:str, prev:Dict={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """

        if query is None or len(query) == 0:
            return []

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        query=query.strip()
        # query = self.tokenize(query)

        
        if("(" in query):
            query=self.solve_parpos(None,0,query)
        
        cont = 0; pos=0; ini=0; field=None

        while(cont<len(query) and '"' in query[cont:]):
            if(cont == 0 and query[cont]!='"'):
                field = cont
            if(query[cont]==' ' and pos==0):
                field=cont+1
            if(query[cont]=='"' and pos==0):
                ini=cont
                pos=1
                if(field is not None and query[cont-1]!=':'):
                    field=None
            elif(query[cont]=='"' and pos==1):
                pos=0
                key = self.hashkey(query[ini+1:cont],cont)
                aux=len(query)
                if(field is not None):
                    self.parpos[key]=self.get_posting(query[ini:cont+1],query[field:ini-1])
                    query=query[:field]+key+query[cont+1:]
                    field=None
                else:
                    self.parpos[key]=self.get_posting(query[ini:cont+1])
                    query=query[:ini]+key+query[cont+1:]                    
                cont-=(aux-(len(query)))

            cont+=1

        que=query.split(' ')
        i = 0

        if(que[i]=='NOT'):
            postinglist=self.calculateposting(que[i+1])
            postinglist = self.reverse_posting(postinglist)
            i+=2
        else:
            postinglist=self.calculateposting(que[i])
            i+=1

        while(i<len(que)):
            if(que[i] not in ['AND','OR']):
                if(que[i]=='NOT'):
                    pos2=self.calculateposting(que[i+1])
                    pos2 = self.reverse_posting(pos2)
                    i+=2
                else:
                    pos2=self.calculateposting(que[i])
                    i+=1
                if(isinstance(postinglist,dict)):
                        postinglist=list(postinglist.keys())
                if(isinstance(pos2,dict)):
                        pos2=list(pos2.keys())

                postinglist=self.and_posting(postinglist,pos2)

            else:    
                aux=i
                if(que[i+1]=='NOT'):
                    pos2=self.calculateposting(que[i+2])
                    pos2 = self.reverse_posting(pos2)
                    i+=3
                else:
                    pos2=self.calculateposting(que[i+1])
                    i+=2

                if(que[aux]=='AND'):
                    postinglist=self.and_posting(postinglist,pos2)

                else:
                    postinglist=self.or_posting(postinglist,pos2)

        return postinglist



    def get_posting(self, term:str, field:Optional[str]=None):
        """

        Devuelve la posting list asociada a un termino. 
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la ampliacion de stemming


        param:  "term": termino del que se debe recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario si se hace la ampliacion de multiples indices

        return: posting list
        
        NECESARIO PARA TODAS LAS VERSIONES

        """
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        pass

        """
            primero comprueba si es una solicitud no válida, es decir, 
            si hay un permuterm (indicado con * o ? dentro de la query)
            dentro de un posicional (envuelto en "")
        """
        if term[0] == '"' and ('*' in term or '?' in term):
            print("No se puede utilizar permuterm dentro del positional.")
            return []
        


        """
            comprueba qué tipo de término es y lo manda a la función correspondiente:
            - si contiene * o ? significa que es permuterm
            - si comienza con " significa que es posicional
            - si está indicado el stemming
            a todos éstos métodos se les envía también los fields indicados, si es una
            query normal:
            - si se le envía field se busca en el índice del field el término
            - si no se envía field se busca el término en el índica 'all'
            si falla el método implica que el término no se ha encontrado en el 
            diccionario y, por tanto, su posting list corresponde a una lista vacía
        """
        try:
            # si tiene comodines usar permuterm
            if '*' in term or '?' in term:
                return self.get_permuterm(term, field)
            # si tiene dobles comillas usar posicionales
            elif(term[0] == '"'):
                return self.get_positionals(term, field)
            elif(self.use_stemming):
                return self.get_stemming(term, field)
            elif(field != None):
                return self.index[field][term]
            else:
                return self.index['all'][term]
        except:
            return []
            

        
        
    
            


    def get_positionals(self, terms:str, field):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        #Retiramos las comillas de la consulta y la dividimos.
        t = terms[1:len(terms)-1].split()
        postinglist=[]
        #Si no hay un campo especificado, se usa el por defecto.
        if(not field):
            field=self.def_field
        #Si solo hay un termino en la consulta se devuelve su diccionario.
        if(len(t)==1):
            return list(self.index[field][t[0]].keys())

        #Por cada aparición del primer termino en cada articulo se comprueba si cada uno de los términos aparece en el artículo y ocupa 
        #su posición correspondiente. En caso de que se llegue al último termino de la consulta y cumpla las condiciones se añade la posición 
        #del primer término de la consulta a la lista del artículo correspondiente en el diccionario que se devolverá cuando acaben las comprobaciones.
        for url in self.index[field][t[0]]:
            noturl=True
            for posicion in self.index[field][t[0]][url]:
                if(noturl):
                    for termino in range(1,len(t)):
                        if(noturl and url in self.index[field][t[termino]]):
                            if((posicion+termino) in self.index[field][t[termino]][url]):
                                if(termino==len(t)-1):
                                    postinglist.append(url)
                                    noturl=False
                            else:
                                break
                        else:
                            break
                else:
                    break
        
        return postinglist




    def get_stemming(self, term:str, field: Optional[str]=None):
        """

        Devuelve la posting list asociada al stem de un termino.
        NECESARIO PARA LA AMPLIACION DE STEMMING

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        #Metodo muy sencillo, pasamos la query por el stemmer y el resultado lo intentamos encontrar en el diccionario.
        stem = self.stemmer.stem(term)
        field = self.def_field if field is None else field
        if(stem in self.sindex[field]):
            #Si encontramos el stem en el diccionario devolvemos la posting list asociada
            return self.sindex[field][stem]
        else:
            #Sino devolvemos una lista vacía
            return []

    def get_permuterm(self, term:str, field:Optional[str]=None):
        """
        Devuelve la posting list asociada a un termino utilizando el indice permuterm.
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list
        """
        #Permutamos la string hasta que la wildcard este al final de la palabra
        field = self.def_field if field is None else field
        perm = "".join([term,'$'])
        while((perm[-1] != "*") and (perm[-1] != "?")):
            perm = "".join([perm[-1:],perm[:-1]])
        simbolo = perm[-1]
        perm = perm[:-1]

        #Busqueda binaria
        inicio = 0
        fin = len(self.ptindex[field])-1
        while(inicio <= fin):
            medio = int((inicio + fin)/2)
            if((self.ptindex[field][medio][0] > perm) or (self.ptindex[field][medio][0].startswith(perm))):
                fin = medio - 1
            else:
                inicio = medio + 1
        #Comprobamos si el indice inicio pertenece a la lista
        if(inicio == len(self.ptindex[field])):
            return []
        #Variable con el resultado final:
        aux = {}
        #Diferenciamos dos casos:
        #En el caso de que la wildcard sea *
        if simbolo == '*':
            #Mientras el permuterm empiece por nuestra query (perm) añadimos la posting list
            while self.ptindex[field][inicio][0].startswith(perm) and inicio < len(self.ptindex[field]):
                aux.update(self.index[field][self.ptindex[field][inicio][1]])
                inicio += 1
        #En el caso de que la wildcard sea ?
        else:       
            longitud = len(perm)
            '''Mientras el permuterm empiece por nuestra query y la longitud del permuterm sea 
            1 mas que la query añadimos la posting list'''
            while self.ptindex[field][inicio][0].startswith(perm) and inicio < len(self.ptindex[field]):
                if (longitud+1 == len(self.ptindex[field][inicio][0])):
                    aux.update(self.index[field][self.ptindex[field][inicio][1]])
                inicio += 1
        #Devolvemos el resultado ordenado        
        return list(sorted(aux))



    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """
        
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        notlist=list()
        pcont=0
        articles=list(self.articles.keys())
        i=0

        while(pcont<len(p)):
            if(p[pcont]>articles[i]):
                notlist.append(articles[i])
                i+=1
            elif(articles[i]>p[pcont]):
                notlist.append(articles[i])
                pcont+=1
            else:
                i+=1
                pcont+=1
        while(i<len(articles)):
            notlist.append(articles[i])
            i+=1


        return notlist


    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Calcula el AND de dos posting list de forma EFICIENTE
        param:  "p1", "p2": posting lists sobre las que calcular
        return: posting list con los artid incluidos en p1 y p2

        """
        
        pass
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

        res = []
        i = 0
        j = 0

        """
            recorre las dos posting list:
                - si el artId es igual, significa que se debe añadir a la respuesta
                y avanzar en ambas posting list
                - si el artId de alguna de las listas es menor que el de la otra,
                avanza en esa lista
            al terminar una de las listas se sabe que lo que queda en la otra no 
            debe ir en la respuesta dado que es un AND
        """
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                res.append(p1[i])
                i += 1; j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j += 1
        return res




    def or_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Calcula el OR de dos posting list de forma EFICIENTE
        param:  "p1", "p2": posting lists sobre las que calcular
        return: posting list con los artid incluidos de p1 o p2

        """

        pass
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

        #### SIN PROBAR 

        res = []
        i = 0
        j = 0

        """
            recorre las dos posting list:
                - si el artId es igual, se añade a la respuesta y avanza en ambas 
                posting lists
                - si el artId de alguna de las listas es menor que el de la otra,
                se añade a la respuesta y avanza en su posting list.
            al terminar una de las listas hay que terminar de añadir lo que hay en
            la otra
        """

        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                res.append(p1[i])
                i += 1; j += 1
            elif p1[i] < p2[j]:
                res.append(p1[i])
                i += 1
            else:
                res.append(p2[j])
                j += 1

        while i < len(p1):
            res.append(p1[i])
            i += 1
        while j < len(p2):
            res.append(p2[j])
            j += 1

        return res

        


    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """

        
        pass
        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################


        #### SIN PROBAR 

        res = []
        i = 0
        j = 0

        """
            recorre las dos posting list:
                - si el artId es igual, avanza en ambas posting lists
                - si el artId de la primera posting list es menor que el de la 
                  segunda, significa que en el artículo está el primer término
                  y no el segundo, por lo que se añade a la respuesta
                - si el artId de la segunda posting list es menor que el de la
                  primera, avanzar una posición
            si termina antes la segunda posting list se termina de recorrer la
            primera sabiendo que todos los artículos tendrán sólo el primer
            término y añadiéndose a la respuesta
        """

        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                i += 1; j += 1
            elif p1[i] < p2[j]:
                res.append(p1[i])
                i += 1
            else:
                j += 1

        while i < len(p1):
            res.append(p1[i])
            i += 1

        return res



    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result = len(self.solve_query(query))
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True                    
            # else:
            #     print(query)
        return not errors


    def solve_and_show(self, query:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados 

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        pass

        ################
        ## COMPLETAR  ##
        ################

        """
            de serie se  muestran sólo los primeros diez artículos recuperados
            para mostrarlos todos se debe especificar
        """

        get = 10
        
        if len(query) > 0 and query[0] != '#':

            # resuelve la query
            res = self.solve_query(query)

            if(self.show_all):
                get = len(res)

            # se muestran snippets 
            if(self.show_snippet and len(self.query_words(query)) < 5):
                # recorre los documentos mencionados en la respuesta
                for i, docId in enumerate(res):
                    if(i >= get): break

                    # obtiene los url y los títulos
                    url = self.articles[docId][0]
                    title = self.articles[docId][1]

                    # abre los documentos usados para hacer el índice 
                    for z in range(1, len(self.docs)):
                        for line in open(f'{self.docs[z]}'):
                            # parsea los artículos usados
                            j = self.parse_article(line)
                            # si el url del artículo es el que se está buscando
                            if(j['url']) == url:

                                # muestra su posición de recuperación, el id del artículo y su url
                                print(f'# {i + 1} ( {docId})\t\u2192 {url}')
                                # muestra el título
                                print(f'# Titulo del articulo: {title}')
                                # para cada palabra de la query
                                for q in self.query_words(query):
                                    
                                    # expresion regular que encuentra una frase con la palabra dada
                                    frase_regex = re.compile(rf'(^|\.\s|\n)([^\.\n]*\b{q}\b[^\.\n]*)(?=\.\s|\n|$)', re.IGNORECASE) 
                                    # saca las frases en las que aparece la palabra
                                    frases = frase_regex.finditer(j['all'])
                                    res_snippets = ''
                                    

                                    # añade reparados entre frases
                                    for f in frases:
                                        res_snippets += f.group(0).replace('\n', '') + ' [...] '
                        
                                    print(res_snippets)

            # no se muestran snippets
            else: 
                # para cada documento recuperado
                for i, artId in enumerate(res):
                    if(i >= get): break
                    # recupera el url y el título
                    url = self.articles[artId][0]
                    title = self.articles[artId][1]
                    # muestra la posición en la recuperación, el id del artíuclo y el título
                    print(f'# {i + 1} ( {artId}) {title}:\t{url}')

            print('=====================================')
            print('Number of results:', len(res))



    def query_words(self, phrase):
        exclude_words = {"AND", "OR", "NOT"}
        words = phrase.split()
        filtered_words = [word for word in words if word not in exclude_words]
        return filtered_words

        

