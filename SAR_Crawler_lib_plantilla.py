#! -*- encoding: utf8 -*-
import heapq as hq

from typing import Tuple, List, Optional, Dict, Union

import requests
import bs4
import re
from urllib.parse import urljoin
import json
import math
import os

class SAR_Wiki_Crawler:

    def __init__(self):
        # Expresión regular para detectar si es un enlace de la Wikipedia
        self.wiki_re = re.compile(r"(http(s)?:\/\/(es)\.wikipedia\.org)?\/wiki\/[\w\/_\(\)\%]+")
        # Expresión regular para limpiar anclas de editar
        self.edit_re = re.compile(r"\[(editar)\]")
        # Formato para cada nivel de sección
        self.section_format = {
            "h1": "##{}##",
            "h2": "=={}==",
            "h3": "--{}--"
        }

        # Expresiones regulares útiles para el parseo del documento
        self.title_sum_re = re.compile(r"##(?P<title>.+)##\n(?P<summary>((?!==.+==).+|\n)+)(?P<rest>(.+|\n)*)")
        self.sections_re = re.compile(r"==.+==\n")
        self.section_re = re.compile(r"==(?P<name>.+)==\n(?P<text>((?!--.+--).+|\n)*)(?P<rest>(.+|\n)*)")
        self.subsections_re = re.compile(r"--.+--\n")
        self.subsection_re = re.compile(r"--(?P<name>.+)--\n(?P<text>(.+|\n)*)")


    def is_valid_url(self, url: str) -> bool:
        """Verifica si es una dirección válida para indexar

        Args:
            url (str): Dirección a verificar

        Returns:
            bool: True si es valida, en caso contrario False
        """
        return self.wiki_re.fullmatch(url) is not None


    def get_wikipedia_entry_content(self, url: str) -> Optional[Tuple[str, List[str]]]:
        """Devuelve el texto en crudo y los enlaces de un artículo de la wikipedia

        Args:
            url (str): Enlace a un artículo de la Wikipedia

        Returns:
            Optional[Tuple[str, List[str]]]: Si es un enlace correcto a un artículo
                de la Wikipedia en inglés o castellano, devolverá el texto y los
                enlaces que contiene la página.

        Raises:
            ValueError: En caso de que no sea un enlace a un artículo de la Wikipedia
                en inglés o español
        """
        if not self.is_valid_url(url):
            raise ValueError((
                f"El enlace '{url}' no es un artículo de la Wikipedia en español"
            ))

        try:
            req = requests.get(url)
        except Exception as ex:
            print(f"ERROR: - {url} - {ex}")
            return None


        # Solo devolvemos el resultado si la petición ha sido correcta
        if req.status_code == 200:
            soup = bs4.BeautifulSoup(req.text, "lxml")
            urls = set()

            for ele in soup.select((
                'div#catlinks, div.printfooter, div.mw-authority-control'
            )):
                ele.decompose()

            # Recogemos todos los enlaces del contenido del artículo
            for a in soup.select("div#bodyContent a", href=True):
                href = a.get("href")
                if href is not None:
                    urls.add(href)

            # Contenido del artículo
            content = soup.select((
                "h1.firstHeading,"
                "div#mw-content-text h2,"
                "div#mw-content-text h3,"
                "div#mw-content-text h4,"
                "div#mw-content-text p,"
                "div#mw-content-text ul,"
                "div#mw-content-text li,"
                "div#mw-content-text span"
            ))

            dedup_content = []
            seen = set()

            for element in content:
                if element in seen:
                    continue

                dedup_content.append(element)

                # Añadimos a vistos, tanto el elemento como sus descendientes
                for desc in element.descendants:
                    seen.add(desc)

                seen.add(element)

            text = "\n".join(
                self.section_format.get(element.name, "{}").format(element.text)
                for element in dedup_content
            )

            # Eliminamos el texto de las anclas de editar
            text = self.edit_re.sub('', text)

            return text, sorted(list(urls))

        return None

    '''
        args: 
            text: es el articulo de wikipedia sin el texto ni el resumen, es decir solo la parte correspondiente a las secciones
    
        returns:
            array con las secciones
    
    '''
    def parse_text_sections(self, text: str) -> Optional[Dict[str, Union[str,List]]]:

        sections = []
        
        #calculamos el numero de secciones (numero de elementos que tendra el iterable)
        num = len(self.sections_re.findall(text))

        #en algunos casos hay documentos que contienen el titulo de una sola seccion sin texto ni contenido
        #en esos casos hacemos return [] para que no de error al hacer el match
        if num == 0:
            return []
    
        #buscamos los matches de los titulos de las secciones para calcular las posiciones
        matches = self.sections_re.finditer(text)

        #recorremos por pares de matches consecutivos, ya que una seccion abarca desde su titulo hasta el de la siguiente -1
        #en el caso del ultimo match va desde su titulo hasta el final del documento

        #primer y segundo elemento de los pares consecutivos del iterable
        first = next(matches)
        second = None

        #recorremos todos los pares del iterable
        while num>0:
            #la primera iteracion first no cambia
            if second is not None:
                first = second
            
            #si no es la ultima sacamos la siguiente
            if num>1:
                second = next(matches)

            #hacemos el match de la expresion regular de la seccion

            #si no es la ultima la aplicamos desde su titulo hasta el titulo de la siguiente -1
            if num>1:
                section_match = self.section_re.search(text[first.span()[0]:second.span()[0]])
            else:
                #si es la ultima seccion va desde el titulo hasta el final
                section_match = self.section_re.search(text[first.span()[0]:])

            subsections = []

            #si tiene subsecciones las parseamos
            if len(section_match.group('rest'))>0:
                subsections = self.parse_section_subsections(section_match.group('rest'))

            #añadimos la entrada al array de seccions
            sections.append({
                'name': section_match.group('name'),
                'text': section_match.group('text'),
                'subsections': subsections
            })

            num -= 1

        return sections
    
    '''
        args: 
            text: texto con solo la parte de las subsecciones de una seccion

        returns:
            array con las subsecciones
    
    '''
    def parse_section_subsections(self, text):

        subsections = []
    
        #buscamos los matches de los titulos de las subsecciones para calcular las posiciones
        matches = self.subsections_re.finditer(text)

        #calculamos el numero de subsecciones (numero de elementos que tendra el iterable)
        num = len(self.subsections_re.findall(text))

        #recorremos por pares de matches consecutivos, ya que una subseccion abarca desde su titulo hasta el de la siguiente -1
        #en el caso del ultimo match va desde su titulo hasta el final del texto

        #primer y segundo elemento de los pares consecutivos del iterable
        first = next(matches)
        second = None

        #recorremos todos los pares del iterable
        while num>0:
            #la primera iteracion first no cambia
            if second is not None:
                first = second
            
            #si no es la ultima sacamos la siguiente
            if num>1:
                second = next(matches)

            #hacemos el match de la expresion regular de la subseccion

            #si no es la ultima la aplicamos desde su titulo hasta el titulo de la siguiente -1
            if num>1:
                subsection_match = self.subsection_re.search(text[first.span()[0]:second.span()[0]])
            else:
                #si es la ultima seccion va desde el titulo hasta el final
                subsection_match = self.subsection_re.search(text[first.span()[0]:])

            #añadimos la entrada al array de subsecciones
            subsections.append({
                'name': subsection_match.group('name'),
                'text': subsection_match.group('text')
            })

            num -= 1

        return subsections

    def parse_wikipedia_textual_content(self, text: str, url: str) -> Optional[Dict[str, Union[str,List]]]:
        """Devuelve una estructura tipo artículo a partir del text en crudo

        Args:
            text (str): Texto en crudo del artículo de la Wikipedia
            url (str): url del artículo, para añadirlo como un campo

        Returns:

            Optional[Dict[str, Union[str,List[Dict[str,Union[str,List[str,str]]]]]]]:

            devuelve un diccionario con las claves 'url', 'title', 'summary', 'sections':
                Los valores asociados a 'url', 'title' y 'summary' son cadenas,
                el valor asociado a 'sections' es una lista de posibles secciones.
                    Cada sección es un diccionario con 'name', 'text' y 'subsections',
                        los valores asociados a 'name' y 'text' son cadenas y,
                        el valor asociado a 'subsections' es una lista de posibles subsecciones
                        en forma de diccionario con 'name' y 'text'.

            en caso de no encontrar título o resúmen del artículo, devolverá None

        """
        def clean_text(txt):
            return '\n'.join(l for l in txt.split('\n') if len(l) > 0)
        
        text = clean_text(text)
        
        document = {}

        match = self.title_sum_re.search(text)

        document['url'] = url
        document['title'] = match.group('title')
        document['summary'] = match.group('summary')

        if document['title'] is None or document['summary'] is None:
            return None

        text = match.group('rest')

        #si tiene secciones
        if len(text)>0:
            document['sections'] = self.parse_text_sections(text)
        else:
            document['sections'] = []

        return document
        

    def save_documents(self,
        documents: List[dict], base_filename: str,
        num_file: Optional[int] = None, total_files: Optional[int] = None
    ):
        """Guarda una lista de documentos (text, url) en un fichero tipo json lines
        (.json). El nombre del fichero se autogenera en base al base_filename,
        el num_file y total_files. Si num_file o total_files es None, entonces el
        fichero de salida es el base_filename.

        Args:
            documents (List[dict]): Lista de documentos.
            base_filename (str): Nombre base del fichero de guardado.
            num_file (Optional[int], optional):
                Posición numérica del fichero a escribir. (None por defecto)
            total_files (Optional[int], optional):
                Cantidad de ficheros que se espera escribir. (None por defecto)
        """
        assert base_filename.endswith(".json")

        if num_file is not None and total_files is not None:
            # Separamos el nombre del fichero y la extensión
            base, ext = os.path.splitext(base_filename)
            # Padding que vamos a tener en los números
            padding = len(str(total_files))

            out_filename = f"{base}_{num_file:0{padding}d}_{total_files}{ext}"

        else:
            out_filename = base_filename

        with open(out_filename, "w", encoding="utf-8", newline="\n") as ofile:
            for doc in documents:
                print(json.dumps(doc, ensure_ascii=True), file=ofile)


    def start_crawling(self, 
                    initial_urls: List[str], document_limit: int,
                    base_filename: str, batch_size: Optional[int], max_depth_level: int,
                    ):        
         

        """Comienza la captura de entradas de la Wikipedia a partir de una lista de urls válidas, 
            termina cuando no hay urls en la cola o llega al máximo de documentos a capturar.
        
        Args:
            initial_urls: Direcciones a artículos de la Wikipedia
            document_limit (int): Máximo número de documentos a capturar
            base_filename (str): Nombre base del fichero de guardado.
            batch_size (Optional[int]): Cada cuantos documentos se guardan en
                fichero. Si se asigna None, se guardará al finalizar la captura.
            max_depth_level (int): Profundidad máxima de captura.
        """
        
        # URLs válidas, ya visitadas (se hayan procesado, o no, correctamente)
        visited = set()
        # URLs en cola
        to_process = set(initial_urls)
        # Direcciones a visitar
        queue = [(0, "", url) for url in to_process]
        hq.heapify(queue)
        # Buffer de documentos capturados
        documents: List[dict] = []
        # Contador del número de documentos capturados
        total_documents_captured = 0
        # Contador del número de ficheros escritos
        files_count = 0
        
        # En caso de que no utilicemos bach_size, asignamos None a total_files
        # así el guardado no modificará el nombre del fichero base
        if batch_size is None:
            total_files = None
        else:
            # Suponemos que vamos a poder alcanzar el límite para la nomenclatura
            # de guardado
            total_files = math.ceil(document_limit / batch_size)

        # COMPLETAR
        while total_documents_captured < document_limit and len(queue)>0:

            #sacamos el siguiente nodo del arbol del heap
            file_node = hq.heappop(queue)

            #sacamos sus caracteristicas
            node_depth = file_node[0]
            node_father = file_node[1]
            node_url = file_node[2]
            
            '''
            Comprueba si la url es valida
            No se comprueba la profundidad porque no se añaden enlaces con mayor profundidad que la maxima 
            y la profundidad de los enlaces iniciales siempre es 0
            '''
            if self.is_valid_url(node_url) and node_url.startswith("http"):
                
                #actualizamos lista de urls visitadas
                visited.add(node_url)

                #el raw content es una tupla con dos elementos, el texto del articulo y la lista con urls citados
                raw_content = self.get_wikipedia_entry_content(node_url)

                doc = self.parse_wikipedia_textual_content(raw_content[0],node_url)
                '''
                print('--------------------------------------------------------------------------------------------')
                print('URL: '+node_url)
                print('depth: ' + str(node_depth))
                print('Father: '+node_father)
                print('--------------------------------------------------------------------------------------------')
                '''
                if doc is not None:
                    documents.append(doc)
                    #tras capturar el documento correctamente actualizamos numero de documentos captuados
                    total_documents_captured+=1

                    if (batch_size is not None) and (len(documents) == batch_size):
                        files_count += 1
                        self.save_documents(documents, base_filename, files_count, total_files) 
                        documents = []

                #si el nodo actual no esta en el maximo nivel de profundidad
                if node_depth<max_depth_level:
                    #para cada url citado
                    for url in raw_content[1]:
                        
                        #si es una url valida a un articulo de la wikipedia
                        if self.is_valid_url(url):
                            #si la url del articulo es relativa la hacemos absoluta
                            url = urljoin(node_url,url)
                            
                            #si no esta en la lista de visitados lo añadimos al heap
                            if url not in visited:
                                #añadimos nuevo nodo al heap
                                # version debugging que guarda el padre del nodo: 
                                #hq.heappush(queue,(node_depth+1,node_url,url)) 
                                hq.heappush(queue,(node_depth+1,'',url))             

        #al acabar el crawling si no se ha especificado un batch size se guardan todos en el mismo documento
        if batch_size is None:
            files_count += 1
            self.save_documents(documents, base_filename)
        #si se ha especificado batch size pero el documents no ha llegado se guardan los que queden
        elif len(documents)>0:
            files_count += 1
            self.save_documents(documents, base_filename, files_count, total_files) 


    def wikipedia_crawling_from_url(self,
        initial_url: str, document_limit: int, base_filename: str,
        batch_size: Optional[int], max_depth_level: int
    ):
        """Captura un conjunto de entradas de la Wikipedia, hasta terminar
        o llegar al máximo de documentos a capturar.
        
        Args:
            initial_url (str): Dirección a un artículo de la Wikipedia
            document_limit (int): Máximo número de documentos a capturar
            base_filename (str): Nombre base del fichero de guardado.
            batch_size (Optional[int]): Cada cuantos documentos se guardan en
                fichero. Si se asigna None, se guardará al finalizar la captura.
            max_depth_level (int): Profundidad máxima de captura.
        """
        if not self.is_valid_url(initial_url) and not initial_url.startswith("/wiki/"):
            raise ValueError(
                "Es necesario partir de un artículo de la Wikipedia en español"
            )

        self.start_crawling(initial_urls=[initial_url], document_limit=document_limit, base_filename=base_filename,
                            batch_size=batch_size, max_depth_level=max_depth_level)



    def wikipedia_crawling_from_url_list(self,
        urls_filename: str, document_limit: int, base_filename: str,
        batch_size: Optional[int]
    ):
        """A partir de un fichero de direcciones, captura todas aquellas que sean
        artículos de la Wikipedia válidos

        Args:
            urls_filename (str): Lista de direcciones
            document_limit (int): Límite máximo de documentos a capturar
            base_filename (str): Nombre base del fichero de guardado.
            batch_size (Optional[int]): Cada cuantos documentos se guardan en
                fichero. Si se asigna None, se guardará al finalizar la captura.

        """

        urls = []
        with open(urls_filename, "r", encoding="utf-8") as ifile:
            for url in ifile:
                url = url.strip()

                # Comprobamos si es una dirección a un artículo de la Wikipedia
                if self.is_valid_url(url):
                    if not url.startswith("http"):
                        raise ValueError(
                            "El fichero debe contener URLs absolutas"
                        )

                    urls.append(url)

        urls = list(set(urls)) # eliminamos posibles duplicados

        self.start_crawling(initial_urls=urls, document_limit=document_limit, base_filename=base_filename,
                            batch_size=batch_size, max_depth_level=0)





if __name__ == "__main__":
    raise Exception(
        "Esto es una librería y no se puede usar como fichero ejecutable"
    )
