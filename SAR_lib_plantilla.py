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
        ("all", True), ("title", True), ("summary", True), ("section-name", True), ('url', False),
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
                        self.docs[len(self.docs) + 1] = fullname
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################

        if(self.stemming):
            self.make_stemming()

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



        for i, line in enumerate(open(filename)):
            j = self.parse_article(line)

        #already_in_index

        #################
        ### COMPLETAR ###
        #################
            artId = 0;
            
            # si el artículo todavía no se ha analizado
            if(j['url'] not in self.urls):
                artId = len(self.articles) + 1
                self.articles[artId] = [j['url'], j['title']]

                self.urls.add(j['url'])

                fields_to_tokenize = ['all']

                if(self.multifield == True):
                    fields_to_tokenize = ['all', 'title', 'summary', 'section-name']



                #si no se quiere usar el índice posicional 
                if(self.positional == False):
                    for field in fields_to_tokenize:
                        tk = self.tokenize(j[field])
                    #para cada token del documento
                        if(field not in self.index):
                            self.index[field] = {}
                        for t in tk:
                            if(t not in self.index[field]):
                                    self.index[field][t] = []
                                    self.index[field][t].append(artId)
                            else:
                                if(artId not in self.index[field][t]):
                                    self.index[field][t].append(artId)   
                    
                    if('url' not in self.index):
                        self.index['url'] = {}
                    else:
                        if(j['url'] not in self.index['url']):
                            self.index['url'][j['url']] = []
                            self.index['url'][j['url']].append(artId)
                    
                #positional

                if(self.positional == True):

                    
                    for field in fields_to_tokenize:
                        tk = self.tokenize(j[field])
                        if(field not in self.index):
                            self.index[field] = {}
                        i = 0
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
                            i+=1

        # print(self.positional['all']['suma'])
            

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
        fields_to_tokenize = ['all']
        if(self.multifield == True):
            fields_to_tokenize = ['all', 'title', 'summary', 'section-name']
        for field,k in fields_to_tokenize:
            self.sindex[field] = {}
            for token in self.index[field]:
                stemtoken = self.stemmer.stem(token)
                if stemtoken not in self.sindex[field]:
                    self.sindex[field][stemtoken] = self.index[field][token]
                else:
                    self.sindex[field][stemtoken] = list(set(self.sindex[field][stemtoken]).union(set(self.index[field][token])))
        pass
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################


    
    def make_permuterm(self):
        """

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE PERMUTERM


        """
        pass
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################




    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Muestra estadisticas de los indices
        
        """
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        print("========================================")
        print(f"Number of indexed files: {len(self.docs)}")
        print("----------------------------------------")
        print(f"Number of indexed articles: {len(self.articles)}")
        print("----------------------------------------")

        print("TOKENS:")
        print('\t# of tokens in "all":', len(self.index["all"]))
        if(self.multifield):
            print(f"\t# of tokens in 'title': {len(self.index['title'])}")
            print(f"\t# of tokens in 'summary': {len(self.index['summary'])}")
            print(f"\t# of tokens in 'section-name': {len(self.index['section-name'])}")
            print(f"\t# of tokens in 'url': {len(self.index['url'])}")


        if(self.permuterm):
            print("----------------------------------------")
            print("PERMUTERMS:")
            print('\t# of permuterms in "all":', len(self.index["all"]))
            if(self.multifield):
                print(f"\t# of permuterms in 'title': {len(self.index['title'])}")
                print(f"\t# of permuterms in 'summary': {len(self.index['summary'])}")
                print(f"\t# of permuterms in 'section-name': {len(self.index['section-name'])}")
                print(f"\t# of permuterms in 'url': {len(self.index['url'])}")

        if(self.stemming):
            print("----------------------------------------")
            print("STEMS:")
            print('\t# of stems in "all":', len(self.index["all"]))
            if(self.multifield):
                print(f"\t# of stems in 'title': {len(self.index['title'])}")
                print(f"\t# of stems in 'summary': {len(self.index['summary'])}")
                print(f"\t# of stems in 'section-name': {len(self.index['section-name'])}")
                print(f"\t# of stems in 'url': {len(self.index['url'])}")

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
                
                aux=f"par{cont}"
                self.parpos[aux]=self.solve_query(query[ini+1:cont])


                return query[:ini]+aux+query[cont+1:]
            else:
                return query
        else:
            return query




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

        # si es positional: recorrer la query, todo aquello que haya entre comillas se trata como positional y hay que mandarlo junto
        cont = 0
        pos=0
        ini=0
        while('"' in query):
            if(query[cont]=='"' and pos==0):
                ini=cont
                pos=1
            elif(query[cont]=='"' and pos==1):
                pos=0
                aux=f"pos{cont}"
                self.parpos[aux]=self.get_posting(query[ini:cont+1])
                query=query[:ini]+aux+query[cont+1:]
            cont+=1


        if("(" in query):
            cont=0
            query=self.solve_parpos(None,cont,query)

        que=query.split(' ')
        i = 0

        if(que[i]=='NOT'):
            if(':' in que[i+1]):
                field,name=que[i+1].split(':')
                postinglist = self.get_posting(name,field)
            else:
                if(que[i+1] in self.parpos):
                    postinglist=self.parpos[que[i+1]]
                else:
                    postinglist = self.get_posting(que[i+1])
            postinglist = self.reverse_posting(postinglist)
            i=i+2
        else:
            if(':' in que[i]):
                field,name=que[i].split(':')
                postinglist = self.get_posting(name,field)
            else:
                if(que[i] in self.parpos):
                    postinglist=self.parpos[que[i]]
                else:
                    postinglist = self.get_posting(que[i])
            i=i+1
        while(i<len(que)):
            aux=i
            if(que[i+1]=='NOT'):
                if(':' in que[i+2]):
                    field,name=que[i+2].split(':')
                    pos2 = self.get_posting(name,field)
                else:
                    if(que[i+2] in self.parpos):
                        pos2=self.parpos[que[i+2]]
                    else:
                        pos2 = self.get_posting(que[i+2])
                pos2 = self.reverse_posting(pos2)
                i=i+3
            else:
                if(':' in que[i+1]):
                    field,name=que[i+1].split(':')
                    pos2 = self.get_posting(name,field)
                else:
                    if(que[i+1] in self.parpos):
                        pos2=self.parpos[que[i+1]]
                    else:
                        pos2 = self.get_posting(que[i+1])
                i=i+2

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
        
        
    
            


    def get_positionals(self, terms:str, field):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        ########################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE POSICIONALES ##
        ########################################################
        t = terms[1:len(terms)-1].split()
        postinglist={}

        if(not field):
            for url in len(self.index[t[0]]):
                for posicion in url:
                    for termino in range(1,len(t)):
                        if(url in self.index[t[termino]]):
                            if((posicion+termino) in self.index[t[termino]][url]):
                                if(termino == len(t)-1):
                                    if(url not in postinglist):
                                        postinglist[url]=[]
                                    postinglist[url].append(posicion)
                            else:
                                break
                        else:
                            break
        else:
            for url in len(self.index[field][t[0]]):
                for posicion in url:
                    for termino in range(1,len(t)):
                        if(url in self.index[field][t[termino]]):
                            if((posicion+termino) in self.index[field][t[termino]][url]):
                                if(termino == len(t)-1):
                                    if(url not in postinglist):
                                        postinglist[url]=[]
                                    postinglist[url].append(posicion)
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
        
        stem = self.stemmer.stem(term)
        field = "all" if field is None else field
        if(stem in self.sindex["all"]):
            return self.sindex["all"][stem]
        else:
            return []
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################

    def get_permuterm(self, term:str, field:Optional[str]=None):
        """

        Devuelve la posting list asociada a un termino utilizando el indice permuterm.
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """

        ##################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA PERMUTERM ##
        ##################################################
        pass



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
        for article in self.articles.keys():
            if article not in p:
                notlist.append(article)
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
            else:
                print(query)
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
        
        if len(query) > 0 and query[0] != '#':
            res = self.solve_query(query)
            
            i = 1
            for docId in res:
                url = self.articles[docId][0]
                title = self.articles[docId][1]

                print(f'# {i} ( {docId}) {title}:\t{url}')
                i += 1

            print('=====================================')
            print('Number of results:', len(res))



        

