import numpy as np
import abc
from   typing import TypeAlias, Callable
import platform
import os
import shutil
import json



class Cache:
    
    __cached : np.ndarray       # loaded from hd numpy 2d matrix
    __index  : int              # index of the cached matrix in the 3d matrix
    __path   : str              # begining of paths to files, containing matrices
    __shape  : tuple[int, int]  # shape of the 2d matrices
    

    def __init__(self, path_start: str, matrix2d_shape: tuple[int, int]):
        self.__path   = path_start
        self.__shape  = matrix2d_shape
        self.__index  = 0
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

    def reload(self) -> None:
        """Reloads cached matrix from disk. Attention: May lead to data loss"""
        self.__load()

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
        try:
            self.__dump()
        except:
            pass

    @property
    def path_start(self) -> str:
        return self.__path

    def remove_from_disk(self) -> None:
        shutil.rmtree(self.__path)



Index    : TypeAlias = tuple[int, int, int]
Matrix2d : TypeAlias = np.ndarray



class Matrix3d(abc.ABC):

    _name   : str
    _shape  : tuple[int, int, int]
    __cache : Cache

    def __init__(self, file_path: str, shape: tuple[int, int, int],
                 cache_shape: tuple[int, int]):
        self.__validate(file_path, shape)
        self._shape = shape
        self.__cache = Cache(file_path, cache_shape)
 
    @abc.abstractmethod
    def _get(self, i: int, j: int, k: int) -> float:
        pass

    @abc.abstractmethod
    def _set(self, i: int, j: int, k: int, item: float) -> None:
        pass

    def __getitem__(self, *index: int) -> float:
        self.__check_index(index[0], self._shape[0])
        if len(index) > 1:
            self.__check_index(index[1], self._shape[1])
            if len(index) > 2:
                self.__check_index(index[2], self._shape[2])
                return self._get(*index)
            else:
                raise IndexError(f"Wrong number of indexes: {len(index)} > 3")





    def __setitem__(self, index: Index, item: float) -> None:
        self._set(*index, item)
    
    def __mul__(self, value: "float | Matrix2d | Matrix3d") -> None:
        self.__operator(value, lambda x, y: x * y, 'multiply')

    def __add__(self, value: "float | Matrix2d | Matrix3d") -> None:
        self.__operator(value, lambda x, y: x + y, 'add')

    def __sub__(self, value: "float | Matrix2d | Matrix3d") -> None:
        self.__operator(value, lambda x, y: x - y, 'substitute')

    def __div__(self, value: "float | Matrix2d | Matrix3d") -> None:
        self.__operator(value, lambda x, y: x / y, 'divide')


    def get_2d_matrix(self, i: int) -> np.ndarray:
        Matrix3d.__check_index(i, self._shape[0])
        return self.__cache[i]

    def fmap(self, function: Callable[[float], float]) -> None:
        (l, m, n) = self._shape
        for i in range(l):
            for j in range(m):
                for k in range(n):
                    # self[(i, j, k)] = function(self[(i, j, k)])
                    self._set(i,j,k, function(self._get(i,j,k)))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def copy(self, name: str) -> "Matrix3d":
        # klass = type(self)
        # new = klass(name, self._shape)
        # n = self._shape[0]
        # for i in range(n):
        #     new.__cache[i] = self.__cache[i]
        # return new
        new = type(self)(name, self._shape)
        shutil.copytree(self.directory, new.directory, dirs_exist_ok=True)
        new.__cache.reload() # otherwise cache will rewrite [0]-matrix with zeros
        return new

    @property
    def directory(self) -> str:
        return self.__cache.path_start[:-1]

    def remove_from_disk(self) -> None:
        self.__cache.remove_from_disk()

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
                    # new[(i, j, k)] = self[(i, j, k)]
                    new._set(i,j,k, self._get(i,j,k))
        return new

    def __operator(self, value: "float | Matrix2d | Matrix3d",
                   operator: Callable[[float, float], float],
                   op_name: str) -> None:
        (l, m, n) = self._shape
        if type(value) == float:
            for i in range(l):
                for j in range(m):
                    for k in range(n):
                        # self[(i,j,j)] = operator(self[(i,j,k)], value)
                        self._set(i,j,k, operator(self._get(i,j,k), value))
        elif type(value) == np.ndarray:
            if len(value.shape) == 1: 
                if l != value.shape[0]:
                    raise ValueError(f"Can't {op_name} matrix {(l, m, n)} by vector {value.shape}")
                for i in range(l):
                    for j in range(m):
                        for k in range(n):
                            # self[(i,j,k)] = operator(self[(i,j,k)], value[i])
                            self._set(i,j,k, operator(self._get(i,j,k), value[i]))
            if len(value.shape) == 2:
                if l != value.shape[0] or m != value.shape[1]:
                    raise ValueError(f"Can't {op_name} matrix {(l, m, n)} by vector {value.shape}")
                for i in range(l):
                    for j in range(m):
                        for k in range(n):
                            # self[(i,j,k)] = operator(self[(i,j,k)], value[i,j])
                            self._set(i,j,k, operator(self._get(i,j,k), value[i, j]))
            if len(value.shape) == 3:
                if self.shape != value.shape:
                    raise ValueError(f"Can't {op_name} matrix {(l, m, n)} by vector {value.shape}")
                for i in range(l):
                    for j in range(m):
                        for k in range(n):
                            # self[(i,j,k)] = operator(self[(i,j,k)], value[i,j,k])
                            self._set(i,j,k, operator(self._get(i,j,k), value[i, j, k]))
            else:
                raise ValueError(f"Can't {op_name} matrix {(l, m, n)} by vector {value.shape}") 
        elif isinstance(value, Matrix3d):
            if (l,m,n) != value.shape:
                raise ValueError(f"Can't {op_name} matrix {(l, m, n)} by vector {value.shape}")
            for i in range(l):
                for j in range(m):
                    for k in range(n):
                        # self[(i,j,k)] = operator(self[(i,j,k)], value[(i,j,k)])
                        self._set(i,j,k, operator(self._get(i,j,k), value._get(i,j,k)))
        else:
            raise ValueError(f"Can't {op_name} Matrix3d by {type(value)}")

    def __validate(self, path: str, shape: tuple[int, int, int]) -> None:
        js_file = path + 'info.json'
        if os.path.exists(path):
            with open(js_file, 'r') as file:
                info = json.load(file)
                message = 'СПАКОЙНА! Спокойно. Ты попытался перезаписать файлы одной матрицы файлами другой'
                message += f'\nУдали папку {path} если ты уверен, что они тебе не нужны'
                assert tuple(info['shape']) == shape, message
        else:
            os.makedirs(path)
            with open(js_file, 'w') as file:
                json.dump({'shape' : list(shape)}, file)    

    @staticmethod
    def __check_index(i: int, up: int) -> None:
        if (i < 0) or (i >= up):
            raise IndexError(f'Index {i} is out of bounds [0..{up}]')



if platform.system() == 'Windows':
    SEP = '\\'
else:
    SEP = '/'



class SlicedBy1(Matrix3d):

    def __init__(self, name: str, shape: tuple[int, int, int]):
        super().__init__(name + SEP + 'by1' + SEP, shape, (shape[1], shape[2]))
    
    def _get(self, i: int, j: int, k: int) -> float:
        return self.__cache[i][j][k]

    def _set(self, i: int, j: int, k: int, item: float) -> float:
        self.__cache[i][j][k] = item



class SlicedBy2(Matrix3d):

    def __init__(self, name: str, shape: tuple[int, int, int]):
        super().__init__(name + SEP + 'by2' + SEP, shape, (shape[0], shape[2]))
    
    def _get(self, i: int, j: int, k: int) -> float:
        return self.__cache[j][i][k]

    def _set(self, i: int, j: int, k: int, item: float) -> float:
        self.__cache[j][i][k] = item



class SlicedBy3(Matrix3d):

    def __init__(self, name: str, shape: tuple[int, int, int]):
        super().__init__(name + SEP + 'by3' + SEP, shape, (shape[0], shape[1]))
    
    def _get(self, i: int, j: int, k: int) -> float:
        return self.__cache[k][i][j]

    def _set(self, i: int, j: int, k: int, item: float) -> float:
        self.__cache[k][i][j] = item


def compress_data(
