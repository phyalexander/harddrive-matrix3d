import numpy as np
import abc
from   typing import TypeAlias, Callable
import platform
import os
import shutil



class GroupCache:
    """ Deprecated as unsuccessfull and not necessary """

    __inner  : list[np.ndarray]
    __fntmpl : str
    __offset : int
    __shape  : tuple[int, int]

    SIZE : int = 50

    def __init__(self, filename_template: str, shape: tuple[int, int]):
        self.__offset = -self.SIZE
        self.__inner = []
        self.__fntmpl = filename_template
        self.__shape = shape

    def __getitem__(self, index: int) -> np.ndarray:
        i = index - self.__offset
        if i < self.SIZE:
            return self.__inner[i]
        else:
            self.__dump()
            n = index // self.SIZE
            self.__offset = n * self.SIZE
            self.__load()
            i = index - self.__offset 
            return self.__inner[i]

    def __setitem__(self, index: int, item: np.ndarray) -> None:
        i = index - self.__offset
        if i < self.SIZE:
            self.__inner[i] = item
        else:
            self.__dump()
            n = index // self.SIZE
            self.__offset = n * self.SIZE
            self.__load()
            i = index - self.__offset 
            self.__inner[i] = item


    def __dump(self) -> None:
        for j in range(min(self.SIZE, len(self.__inner))):
            fname = self.__fntmpl + str(j + self.__offset)
            np.save(fname, self.__inner[j])

    def __load(self) -> None:
        for j in range(self.SIZE):
            fname = self.__fntmpl + str(j + self.__offset)
            try:
                self.__inner[j] = np.load(fname + '.npy')
            except FileNotFoundError:
                self.__inner[j] = np.zeros(self.__shape) # проблема...

    def __del__(self) -> None:
        self.__dump() # to not loose the changes
        

class Cache:
    
    __cached : np.ndarray       # loaded from hd numpy 2d matrix
    __index  : int              # index of the cached matrix in the 3d matrix
    __path   : str              # begining of paths to files, containing matrices
    __shape  : tuple[int, int]  # shape of the 2d matrices
    

    def __init__(self, path_start: str, matrix2d_shape: tuple[int, int]):
        self.__path   = path_start
        self.__shape  = matrix2d_shape
        self.__index  = 0
        self.__create_directory()
        self.__load()

    def __getitem__(self, index: int) -> np.ndarray:
        if self.__index != index:
            self.__dump()
            self.__index = index
            self.__load()
        return self.__cached

    def __setitem__(self, index: int, item: np.ndarray) -> None:
        if self.__index != index:
            self.__dump()
            self.__index = index
        self.__cached = item

    def __dump(self) -> None:
        path = self.__path + str(self.__index)
        np.save(path, self.__cached)

    def __load(self) -> None:
        path = self.__path + str(self.__index) + '.npy'
        try:
            self.__cached = np.load(path)
        except FileNotFoundError:
            self.__cached = np.zeros(self.__shape)
    
    def __del__(self) -> None:
        self.__dump()

    def __create_directory(self) -> None:
        os.makedirs(self.__path)

    def remove_from_disk(self) -> None:
        shutil.rmtree(self.__path)



Index    : TypeAlias = tuple[int, int, int]
Matrix2d : TypeAlias = np.ndarray



class Matrix3d(abc.ABC):

    _name  : str
    _shape : tuple[int, int, int]
    _cache : Cache

    def __init__(self, file_path: str, shape: tuple[int, int, int],
                 cache_shape: tuple[int, int]):
        self._file_path = file_path
        self._shape = shape
        self._cache = Cache(file_path, cache_shape)
 
    @abc.abstractmethod
    def __getitem__(self, index: Index) -> float:
        pass

    @abc.abstractmethod
    def __setitem__(self, index: Index, item: float) -> None:
        pass
    
    def __mul__(self, value: "float | Matrix2d | Matrix3d") -> None:
        self.__operator(value, lambda x, y: x * y, 'multiply')


    def __add__(self, value: "float | Matrix2d | Matrix3d") -> None:
        self.__operator(value, lambda x, y: x + y, 'add')


    def __sub__(self, value: "float | Matrix2d | Matrix3d") -> None:
        self.__operator(value, lambda x, y: x - y, 'substitute')


    def __div__(self, value: "float | Matrix2d | Matrix3d") -> None:
        self.__operator(value, lambda x, y: x / y, 'divide')


    def __operator(self, value: "float | Matrix2d | Matrix3d",
                   operator: Callable[[float, float], float],
                   op_name: str) -> None:
        (l, m, n) = self._shape
        if type(value) == float:
            for i in range(l):
                for j in range(m):
                    for k in range(n):
                        self[(i,j,j)] = operator(self[(i,j,k)], value)
        elif type(value) == np.ndarray:
            if len(value.shape) == 1: 
                if l != value.shape[0]:
                    raise ValueError(f"Can't {op_name} matrix {(l, m, n)} by vector {value.shape}")
                for i in range(l):
                    for j in range(m):
                        for k in range(n):
                            self[(i,j,k)] = operator(self[(i,j,k)], value[i])
            if len(value.shape) == 2:
                if l != value.shape[0] or m != value.shape[1]:
                    raise ValueError(f"Can't {op_name} matrix {(l, m, n)} by vector {value.shape}")
                for i in range(l):
                    for j in range(m):
                        for k in range(n):
                            self[(i,j,k)] = operator(self[(i,j,k)], value[i,j])
            if len(value.shape) == 3:
                if self.shape != value.shape:
                    raise ValueError(f"Can't {op_name} matrix {(l, m, n)} by vector {value.shape}")
                for i in range(l):
                    for j in range(m):
                        for k in range(n):
                            self[(i,j,k)] = operator(self[(i,j,k)], value[i,j,k])
            else:
                raise ValueError(f"Can't {op_name} matrix {(l, m, n)} by vector {value.shape}") 
        elif isinstance(value, Matrix3d):
            if (l,m,n) != value.shape:
                raise ValueError(f"Can't {op_name} matrix {(l, m, n)} by vector {value.shape}")
            for i in range(l):
                for j in range(m):
                    for k in range(n):
                        self[(i,j,k)] = operator(self[(i,j,k)], value[(i,j,k)])
        else:
            raise ValueError(f"Can't {op_name} Matrix3d by {type(value)}")


    def fmap(self, function: Callable[[float], float]) -> None:
        (l, m, n) = self._shape
        for i in range(l):
            for j in range(m):
                for k in range(n):
                    self[(i, j, k)] = function(self[(i, j, k)])


    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def copy(self, name: str) -> "Matrix3d":
        klass = type(self)
        print("klass=type(self) while copy calling", klass)
        new = klass(name, self._shape)
        n = self._shape[0]
        for i in range(n):
            new._cache[i] = self._cache[i]
        return new

    @property
    def file_path(self) -> str:
        return self._file_path

    def remove_from_disk(self) -> None:
        self._cache.remove_from_disk()

    def reslice_by(self, dim: int, name: str) -> "Matrix3d":
        if dim == 1:
            nova_class = SlicedBy1
        elif dim == 2:
            nova_class = SlicedBy2
        elif dim == 3:
            nova_class = SlicedBy3
        else:
            raise ValueError(f"Wrong dimension: {dim}. Can be only 1, 2 or 3")

        if type(self) == nova_class:
            return self.copy(name)
        new = nova_class(name, self._shape)
        l, m, n = self._shape
        for i in range(l):
            for j in range(m):
                for k in range(n):
                    new[(i, j, k)] = self[(i, j, k)]
        return new



if platform.system() == 'Windows':
    SEP = '\\'
else:
    SEP = '/'



class SlicedBy1(Matrix3d):

    def __init__(self, name: str, shape: tuple[int, int, int]):
        super().__init__(name + SEP + 'by1' + SEP, shape, (shape[1], shape[2]))
    
    def __getitem__(self, index: Index) -> float:
        i, j, k = index
        return self._cache[i][j][k]

    def __setitem__(self, index: Index, item: float) -> None:
        i, j, k = index
        self._cache[i][j][k] = item


class SlicedBy2(Matrix3d):

    def __init__(self, name: str, shape: tuple[int, int, int]):
        super().__init__(name + SEP + 'by2' + SEP, shape, (shape[0], shape[2]))
    
    def __getitem__(self, index: Index) -> float:
        i, j, k = index
        return self._cache[j][i][k]

    def __setitem__(self, index: Index, item: float) -> None:
        i, j, k = index
        self._cache[j][i][k] = item


class SlicedBy3(Matrix3d):

    def __init__(self, name: str, shape: tuple[int, int, int]):
        super().__init__(name + SEP + 'by3' + SEP, shape, (shape[0], shape[1]))
    
    def __getitem__(self, index: Index) -> float:
        i, j, k = index
        return self._cache[k][i][j]

    def __setitem__(self, index: Index, item: float) -> None:
        i, j, k = index
        self._cache[k][i][j] = item





