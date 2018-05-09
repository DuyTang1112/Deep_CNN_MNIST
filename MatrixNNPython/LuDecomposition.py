from Matrix import *
class LuDecomposition(object):
    
    def __init__(self,value):
        """Construct a LU decomposition"""
        self.LU=Matrix(0,0)
        if value==None:
            raise ValueError("value")
        self.LU=value.clone()
        lu=self.LU.Array
        rows=value.Rows
        columns=value.Columns
        self.pivotVector=[0 for _ in range(rows)]
        for i in range(rows):
            self.pivotVector[i]=i
        self.pivotSign=1
        LUrowi=[]
        LUcolj=[0 for _ in range(rows)]
        #outer loop
        for j in range(columns):
            #Make a copy of the j-th column to localize references.
            for i in range(rows):
                LUcolj[i] = lu[i][j]
            #apply previous transformations
            for i in range(rows):
                LUrowi=lu[i]
                #Most of the time is spent in the following dot product
                kmax=min(i,j)
                s=0
                for k in range(kmax):
                    s+=LUrowi[k]*LUcolj[k]
                LUcolj[i]-=s 
                LUrowi[j]=LUcolj[i]
            #Find pivot and exchange if necessary.
            p=j
            for i in range(j+1,rows):
                if abs(LUcolj[i])>abs(LUcolj[p]):
                    p=i
            if p!=j:
                for k in range(columns):
                    lu[p][k],lu[j][k]=lu[j][k],lu[p][k]
                self.pivotVector[p],self.pivotVector[j]=self.pivotVector[j],self.pivotVector[p]
                self.pivotSign=-self.pivotSign
            #compute multipliers
            if j<rows and lu[i][j]!=0.0:
                for i in range(j+1,rows):
                    lu[i][j] /= lu[j][j]
        pass

    @property
    def NonSingular(self):
        """Returns if the matrix is non-singular"""
        for j in range(self.LU.Columns):
            if self.LU[j][j] ==0:
                return False
        return True

    @property
    def Determinant(self):
        """Returns the determinant of the matrix"""
        if self.LU.Rows!= self.LU.Columns:
            raise ValueError("Matrix must be square.")
        determinant=self.pivotSign
        for j in range(self.LU.Columns):
            determinant *= self.LU[j][j]
        return determinant

    @property
    def LowerTriangularFactor(self):
        """Returns the lower triangular factor """
        rows=self.LU.Rows
        columns=self.LU.Columns
        X=Matrix(rows,columns)
        for i in range(rows):
            for j in range(columns):
                if i>j:
                    X[i][j]=self.LU[i][j]
                elif i==j:
                    X[i][j]=1.0
                else:
                    X[i][j]=0.0
        return X
    
    @property
    def UpperTriangularFactor(self):
        """Returns the upper triangular factor"""
        rows=self.LU.Rows
        columns=self.LU.Columns
        X=Matrix(rows,columns)
        for i in range(rows):
            for j in range(columns):
                if i<=j:
                    X[i][j]=self.LU[i][j]
                else:
                    X[i][j]=0.0
        return X