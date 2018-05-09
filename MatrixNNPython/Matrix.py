import random
import math
class Matrix(object):    
    def __init__(self,rows,col,value=None):
        """Constructs an empty matrix of the given size
        and assigns a given value to all diagonal elements
        or construct from given array"""
        self.__data=[] #2d array
        self.rows=rows
        self.columns=col
        self.__data=[[0 for _ in range(col)] for _ in range(rows)]
        if value!=None:
            if type(value) is not list:#if value is a default value
                self.__data=[[value for _ in range(col)] for _ in range(rows)]
            else:# if value is a matrix
                for row in value:
                    if len(row)!=len(value[0]):
                        raise ValueError("Argument out of range.")
                self.__data=value
                pass
    
    def __getitem__(self, index):
        return self.__data[index]
    @property 
    def Array(self):
        return self.__data
    @property 
    def Rows(self):
        return self.rows
    @property 
    def Columns(self):
        return self.columns
    @staticmethod
    def Equals(self,right):
        """Determines weather two instances are equal"""
        if self is right:
            return True
        if right==None or self==None :
            return False
        if self.rows!=right.rows or self.columns!=right.columns:
            return False
        for i in range(self.rows):
            for j in range(self.columns):
                if self[i][j]!=right[i][j]:
                    return False
        return True

    def __hash__(self):
        return self.rows+self.columns
    @property
    def Square(self):
        """Return True if the matrix is a square matrix"""
        return self.rows==self.columns
    @property
    def Symmetric(self):
        """Return True if matrix is if the matrix is symmetric"""
        if self.isSquare():
            for i in range(self.rows):
                for j in range(i+1):
                    if self.__data[i][j]!=self.__data[j][i]:
                        return False
            return True
        return False
    def Submatrix(self,startRow,endRow,startColumn,endColumn):
        """Returns a sub matrix extracted from the current matrix.
        startRow: Start row index
		endRow: End row index
		startColumn: Start column index
		endColumn: End column index"""
        if startRow>endRow or startColumn>endColumn or startRow < 0 or startRow >= self.rows or  endRow < 0 or endRow >= self.rows or  startColumn < 0 or startColumn >= self.columns or  endColumn < 0 or endColumn >= self.columns:
            raise ValueError("Argument out of range.")
        X=Matrix(endRow - startRow + 1, endColumn - startColumn + 1)
        x=X.Array
        for i in range(startRow,endRow+1):
            for j in range(startColumn,endColumn+1):
                x[i - startRow][j - startColumn] = self.__data[i][j]
        return X

    def Submatrix_list(self,rowIndexes,columnIndexes):
        """Returns a sub matrix extracted from the current matrix
		rowIndexes: Array of row indices
		columnIndexes: Array of column indices"""
        X=Matrix(len(rowIndexes),len(columnIndexes))
        x=X.Array
        for i in range(len(rowIndexes)):
            for j in range(len(columnIndexes)):
                if rowIndexes[i]<0 or rowIndexes[i] >= self.rows or columnIndexes[j] < 0 or columnIndexes[j] >= self.columns:
                    raise OverflowError("Argument out of range.")
                x[i][j] = self.__data[rowIndexes[i]][columnIndexes[j]]
        return X

    def Submatrix_row(self,i0,i1,c):
        """Returns a sub matrix extracted from the current matrix.
		i0: Starttial row index
		i1: End row index
		c: Array of row indices"""
        if i0 > i1 or i0 < 0 or i0 >= self.rows or i1 < 0 or i1 >= self.rows:
            raise OverflowError("Argument out of range.")
        X= Matrix(i1 - i0 + 1, c.Length)
        x = X.Array
        for i in range(i0,i1+1):
            for j in range(0,len(c)):
                if c[j] < 0 or c[j] >= self.columns: 
                    raise OverflowError("Argument out of range.")
                x[i - i0][j] = self.__data[i][c[j]];
        return X
    def Submatrix_col(self,r,j0,j1):
        """Returns a sub matrix extracted from the current matrix.
		r: Array of row indices
		j0: Start column index
		j1: End column index"""
        if j0 > j1 or j0 < 0 or (j0 >= self.columns) or (j1 < 0) or j1 >= self.columns:
            raise OverflowError("Argument out of range.")
        X=Matrix(len(r), j1-j0+1)
        x=X.Array
        for i in range(len(r)):
            for j in range(j0,j1+1):
                if ((r[i] < 0) or (r[i] >= self.rows)):
                    raise OverflowError("Argument out of range.")
                x[i][j - j0] = self.__data[r[i]][j]
        return X
    def clone(self):
        """Creates a copy of the matrix"""
        X=Matrix(self.rows,self.columns)
        x=X.Array
        for i in range(self.rows):
            for j in range(self.columns):
                x[i][j]=self.__data[i][j]
        return X
    def Tranpose(self):
        """Returns the transposed matrix"""
        X=Matrix(self.columns,self.rows)
        x=X.Array
        for i in range(self.rows):
            for j in range(self.columns):
                x[j][i]= self.__data[i][j]
        return X
    def Absol(self):
        X=Matrix(self.columns,self.rows)
        x=X.Array
        for i in range(self.rows):
            for j in range(self.columns):
                if self.__data[i][j]>0:
                    x[i][j]=1
                if self.__data[i][j]<0:
                    x[j][i]= -1
        return X
    @property
    def Norm1(self):
        """Returns the One Norm for the matrix.
        The maximum column sum."""
        f=0
        for j in range(self.columns):
            s=0
            for i in range(self.rows):
                s+=abs(self.__data[i][j])
            f=max(f,s)
        return f
    @property 
    def InfinityNorm(self):
        """Returns the Infinity Norm for the matrix.
        The maximum row sum."""
        f=0
        for i in range(self.rows):
            s=0
            for j in range(self.columns):
                s+=abs(self.__data[i][j])
            f=max(f,s)
        return f
    @property
    def FrobeniusNorm(self):
        """Returns the Frobenius Norm for the matrix.
        The square root of sum of squares of all elements."""
        f=0
        for i in range(self.rows):
            s=0
            for j in range(self.columns):
                f=self.Hypotenuse(f,data[i][j])
        return f

    @staticmethod
    def Hypotenuse(a,b):
        if abs(a)>abs(b):
            r=b/a
            return abs(a)*math.sqrt(1+r*r)
        if b!=0:
            r=a/b
            return abs(b)*math.sqrt(1+r*r)
        return 0.0
    @staticmethod
    def Negate(self):
        """Unary minus"""
        if self==None:
            raise ValueError("value")
        rows=self.rows
        columns=self.columns
        data=self.__data
        X=Matrix(rows,columns)
        x=X.Array
        for i in range(rows):
            for j in range(columns):
                x[i][j] = -data[i][j]
        return X

    def __neg__(self):
        """Unary minus"""
        if self==None:
            raise ValueError("value")
        return Matrix.Negate(self)

    def __eq__(self, other):
        """Matrix equality."""
        return Matrix.Equals(self,other)

    def __ne__(self, other):
        """Matrix inequality."""
        return not self.Equals(other)
    @staticmethod
    def Add(left,right):
        """Matrix Addition"""
        if left==None:
            raise ValueError("left")
        if right==None:
            raise ValueError("right")
        rows=left.rows
        columns=left.columns
        data=left.Array
        if rows!=right.rows or columns!=right.columns:
            raise ValueError("Matrix dimension do not match.")
        X=Matrix(rows,columns)
        x=X.Array
        for i in range(rows):
            for j in range(columns):
                x[i][j] = data[i][j] + right[i][j]
        return X

    def __add__(self,other):
        """Matrix Addition"""
        if self==None:
            raise ValueError("left")
        if self==None:
            raise ValueError("right")
        return Matrix.Add(self,other)

    @staticmethod
    def Subtract(left,right):
        """Matrix Subtraction"""
        if left==None:
            raise ValueError("left")
        if right==None:
            raise ValueError("right")
        rows=left.rows
        columns=left.columns
        data=left.Array
        if rows!=right.rows or columns!=right.columns:
            raise ValueError("Matrix dimension do not match.")
        X=Matrix(rows,columns)
        x=X.Array
        for i in range(rows):
            for j in range(columns):
                x[i][j] = data[i][j] - right[i][j]
        return X

    def __sub__(self,other):
        """Matrix Subtraction"""
        if self==None:
            raise ValueError("left")
        if self==None:
            raise ValueError("right")
        return Matrix.Subtract(self,other)

    @staticmethod
    def Multiply(left,right):
        """Matrix-scalar or matrix-matrix multiplication"""
        if left==None:
            raise ValueError("left")
        if right==None:
            raise ValueError("right")
        if type(right) is float or type(right) is int:#matrix-scalar
            rows=left.rows
            columns=left.columns
            data=left.Array
            X=Matrix(rows,columns)
            x=X.Array
            for i in range(rows):
                for j in range(columns):
                    x[i][j]=data[i][j]*right
            return X
        else:#matrix-matrix
            rows=left.rows
            data=left.Array
            if right.rows!=left.columns:
                raise ValueError("Matrix dimensions are not valid.")
            columns=right.columns
            X=Matrix(rows,columns)
            x=X.Array
            size=left.columns
            column=[0 for _ in range(size)]
            for j in range(columns):
                for k in range(size):
                    column[k]=right[k][j]#get the column vector at column j 
                for i in range(rows):
                    row=data[i]
                    s=0
                    for m in range(size):
                        s+=row[m]*column[m]
                    x[i][j]=s
            return X

    def __mul__(self,other):
        if self==None:
            raise ValueError("left")
        return Matrix.Multiply(self,other)

    def Solve(self,rightHandSize):
        """Returns the LHS solution vetor if the matrix is square or the least squares solution otherwise."""
        pass

    @property 
    def Inverse(self):
        """Inverse of the matrix if matrix is square, pseudoinverse otherwise."""
        pass

    @property
    def Determinant(self):
        """Determinant if matrix is square"""
        pass

    @property
    def Trace(self):
        """Returns the trace of the matrix.
        Sum of the diagonal elements."""
        trace=0
        for i in range(min(self.rows,self.columns)):
            trace+=self.__data[i][j]
        return trace

    @staticmethod
    def Random(rows,columns):
        """Returns a matrix filled with random values"""
        X=Matrix(rows,columns)
        x=X.Array
        for i in range(rows):
            for j in range(columns):
                x[i][j]=random.random()
        return X

    @staticmethod 
    def Diagonal(rows,columns,value):
        """Returns a diagonal matrix of the given size"""
        X=Matrix(rows,columns)
        x=X.Array
        for i in range(rows):
            for j in range(columns):
                x[i][j]=value if i==j else 0.0
        return X

    def __str__(self):
        """Returns the matrix in a textual form"""
        str=""
        for i in range(self.rows):
            for j in range(self.columns):
                str+="{} ".format(self.__data[i][j])
            str+="\n"
        return str