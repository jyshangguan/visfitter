import numpy as np
from collections import OrderedDict
import model_functions
import matplotlib.pyplot as plt

pi = np.pi

class ModelFunction(object):
    """
    The functions of each model component.  The model is multiplicative if
    multiList is used.

    Parameters
    ----------
    function : string
        The variable name of the function.
    xName : list
        The list of the names of the active variable.
    parFitDict : dict
        The name of the fitting variables.
    parAddDict : dict
        The name of the additional variables for this model.
    multiList (optional): list
        If provided, the model is multiplicative. The model will be
        multiplied to the models in the list.  Otherwise, the model will be
        added with other models that are not multiplicative.

    Notes
    ----
    Revised by SGJY at Jan. 5, 2018 in KIAA-PKU.
    """
    def __init__(self, function, xName, parFitDict={}, parAddDict={}, multiList=None):
        self.__function = function
        self.xName = xName
        self.nx = len(xName)
        self.parFitDict = parFitDict
        self.parAddDict = parAddDict
        self.multiList  = multiList

    def __call__(self, *xpars):
        """
        Call the function with new input x parameters.  The order of the x parameters
        should be consistent with self.xName.
        """
        # The number of input parameters should be the same as the number of x parameters.
        assert len(xpars) == self.nx
        kwargs = {}
        #Add in the parameters for fit
        for loop in range(self.nx):
            kwargs[self.xName[loop]] = xpars[loop]
        for parName in self.parFitDict.keys():
            kwargs[parName] = self.parFitDict[parName]["value"]
        for parName in self.parAddDict.keys():
            kwargs[parName] = self.parAddDict[parName]
        exec "y = model_functions.{0}(**kwargs)".format(self.__function)
        return y

    def if_Add(self):
        """
        Check whether the function is to add or multiply.
        """
        if self.multiList is None:
            return True
        else:
            return False

    def get_function_name(self):
        """
        Get the function name.
        """
        return self.__function

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

class ModelCombiner(object):
    """
    The object to combine all the model components.

    Parameters
    ----------
    modelDict : dict (better to be ordered dict)
        The dict containing all the model functions.
    xList : list
        The list of the default active variable.
    QuietMode : bool
        Use verbose mode if True.

    Notes
    -----
    None.
    """
    def __init__(self, modelDict, xList, dtype=np.float_, QuietMode=True):
        self.__modelDict = modelDict
        self._modelList = modelDict.keys()
        self.__x = xList
        self.__nx = len(xList) # The number of parameters
        self.__xshape = xList[0].shape # The shape of the input parameter.
        self.dtype = dtype
        self._mltList = [] # The list of models to multiply to other models
        self._addList = [] # The list of models to add together
        for modelName in self._modelList:
            mf = modelDict[modelName]
            assert mf.nx == self.__nx # The models should have the same number of x parameters.
            if mf.if_Add():
                self._addList.append(modelName)
            else:
                self._mltList.append(modelName)
        if not QuietMode:
            print "Add: {0}".format(self._addList)
            print "Multiply: {0}".format(self._mltList)

    def get_xList(self):
        """
        Get the array of the default active variable.

        Parameters
        ----------
        None.

        Returns
        -------
        The array of the default active variable.
        """
        return self.__x

    def get_nx(self):
        """
        Get the number of x parameters.
        """
        return self.__nx

    def get_xshape(self):
        """
        Get the shape of x parameters.
        """
        return self.__xshape

    def set_xList(self, *xList):
        """
        Reset the default active variable array.

        Parameters
        ----------
        xList : array like
            The new array of the active variable.

        Returns
        -------
        None.
        """
        assert len(xList) == self.__nx
        self.__x = xList
        self.__xshape = xList[0].shape # Reset the shape of the input.

    def combineResult(self, *xpars):
        """
        Return the model result combining all the components.

        Parameters
        ----------
        *xpars : list
            The list of active variable of the models.  The inputs should be
            numpy arrays.

        Returns
        -------
        result : array like
            The result of the models all combined.

        Notes
        -----
        None.
        """
        if len(xpars) == 0:
            xpars = self.__x
            xshape = self.__xshape
        else:
            assert len(xpars) == self.__nx # The input parameter number should be consistent.
            xshape = xpars[0].shape
        #-> Calculate the add model components
        addCmpDict = {}
        for modelName in self._addList:
            mf = self.__modelDict[modelName]
            addCmpDict[modelName] = mf(*xpars)
        #-> Manipulate the model components
        for modelName in self._mltList:
            mf = self.__modelDict[modelName]
            my = mf(*xpars) # multiplied y component
            #--> Multiply the current component to the target models
            for tmn in mf.multiList:
                addCmpDict[tmn] *= my
        #-> Add up all the add models
        result = np.zeros(xshape, dtype=self.dtype)
        #print addCmpDict
        for modelName in self._addList:
            result += addCmpDict[modelName]
        return result

    def componentResult(self, *xpars):
        """
        Return the results of all the add components multiplied by the
        multiplicative models correspondingly.

        Parameters
        ----------
        *xpars : list
            The list of active variable of the models.

        Returns
        -------
        result : ordered dict
            The result of the model components.

        Notes
        -----
        None.
        """
        if len(xpars) == 0:
            xpars = self.__x
        else:
            assert len(xpars) == self.__nx # The input parameter number should be consistent.
        #-> Calculate the add model components
        result = OrderedDict()
        for modelName in self._addList:
            mf = self.__modelDict[modelName]
            result[modelName] = mf(*xpars)
        #-> Manipulate the model components
        for modelName in self._mltList:
            mf = self.__modelDict[modelName]
            my = mf(*xpars) # multiplied y component
            #--> Multiply the current component to the target models
            for tmn in mf.multiList:
                result[tmn] *= my
        return result

    def componentAddResult(self, *xpars):
        """
        Return the original results of add models without multiplied other models.

        Parameters
        ----------
        *xpars : list
            The list of active variable of the models.

        Returns
        -------
        result : ordered dict
            The result of the model components.

        Notes
        -----
        None.
        """
        if len(xpars) == 0:
            xpars = self.__x
        else:
            assert len(xpars) == self.__nx # The input parameter number should be consistent.
        result = OrderedDict()
        for modelName in self._addList:
            result[modelName] = self.__modelDict[modelName](*xpars)
        return result

    def componentMltResult(self, *xpars):
        """
        Return the original results of multiplicative models.


        Parameters
        ----------
        *xpars : list
            The list of active variable of the models.

        Returns
        -------
        result : ordered dict
            The result of the model components.

        Notes
        -----
        None.
        """
        if len(xpars) == 0:
            xpars = self.__x
        else:
            assert len(xpars) == self.__nx # The input parameter number should be consistent.
        result = OrderedDict()
        for modelName in self._mltList:
            result[modelName] = self.__modelDict[modelName](*xpars)
        return result

    def get_modelDict(self):
        """
        Get the dict of all the models.
        """
        return self.__modelDict

    def get_modelAddList(self):
        """
        Get the name list of the add models.
        """
        return self._addList

    def get_modelMltList(self):
        """
        Get the name list of the multiply models.
        """
        return self._mltList

    def get_modelParDict(self):
        modelParDict = OrderedDict()
        for modelName in self._modelList:
            model = self.__modelDict[modelName]
            modelParDict[modelName] = model.parFitDict
        return modelParDict

    def get_parList(self):
        """
        Return the total number of the fit parameters.
        """
        parList = []
        for modelName in self._modelList:
            model = self.__modelDict[modelName]
            modelParDict = model.parFitDict
            for parName in modelParDict.keys():
                parList.append(modelParDict[parName]["value"])
        return parList

    def get_parVaryList(self):
        """
        Return the total number of the fit parameters that can vary.
        """
        parList = []
        for modelName in self._modelList:
            model = self.__modelDict[modelName]
            modelParDict = model.parFitDict
            for parName in modelParDict.keys():
                if modelParDict[parName]["vary"]:
                    parList.append(modelParDict[parName]["value"])
                else:
                    pass
        return parList

    def get_parVaryRanges(self):
        """
        Return a list of ranges for all the variable parameters.
        """
        parRList = []
        for modelName in self._modelList:
            model = self.__modelDict[modelName]
            modelParDict = model.parFitDict
            for parName in modelParDict.keys():
                if modelParDict[parName]["vary"]:
                    parRList.append(modelParDict[parName]["range"])
                else:
                    pass
        return parRList

    def get_parVaryNames(self, latex=True):
        """
        Return a list of names for all the variable parameters. The latex format
        is preferred. If the latex format is not found, the variable name is used.
        """
        parNList = []
        for modelName in self._modelList:
            model = self.__modelDict[modelName]
            modelParDict = model.parFitDict
            for parName in modelParDict.keys():
                if modelParDict[parName]["vary"]:
                    if latex:
                        name = modelParDict[parName].get("latex", parName)
                    else:
                        name = parName
                    parNList.append(name)
                else:
                    pass
        return parNList

    def updateParFit(self, modelName, parName, parValue, QuietMode=True):
        model = self.__modelDict[modelName]
        if not QuietMode:
            orgValue = model.parFitDict[parName]
            print "[{0}][{1}] {2}->{3}".format(modelName, parName, orgValue, parValue)
        if model.parFitDict[parName]["vary"]:
            model.parFitDict[parName]["value"] = parValue
        else:
            raise RuntimeError("[ModelCombiner]: {0}-{1} is fixed!".format(modelName, parName))

    def updateParList(self, parList):
        """
        Updata the fit parameters from a list.
        """
        counter = 0
        for modelName in self._modelList:
            model = self.__modelDict[modelName]
            modelParDict = model.parFitDict
            for parName in modelParDict.keys():
                if modelParDict[parName]["vary"]:
                    modelParDict[parName]["value"] = parList[counter]
                    counter += 1
                else:
                    pass

    def updateParAdd(self, modelName, parName, parValue, QuietMode=True):
        model = self.__modelDict[modelName]
        if not QuietMode:
            orgValue = model.parAddDict[parName]
            print "[{0}][{1}] {2}->{3}".format(modelName, parName, orgValue, parValue)
        model.parAddDict[parName] = parValue

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

class VisModel(ModelCombiner):
    def __init__(self, input_model_dict, func_lib, x_list,
                 par_add_dict_all={}, QuietMode=False, **kwargs):
        """
        Generate the ModelClass object from the input model dict.

        Parameters
        ----------
        input_model_dict : dict (better to be ordered dict)
            The dict of input model informations.
            An example of the format of the dict elements:
                "Slope": {                    # The name of the model is arbitrary.
                    "function": "Linear"      # Necessary to be exactly the same as
                                              # the name of the variable.
                    "a": {                    # The name of the first parameter.
                        "value": 3.,          # The value of the parameter.
                        "range": [-10., 10.], # The prior range of the parameter.
                        "type": "c",          # The type (continuous/discrete) of
                                              # the parameter. Currently, it does
                                              # not matter...
                        "vary": True,         # The toggle whether the parameter is
                                              # fixed (if False).
                        "latex": "$a$",       # The format for plotting.
                    }
                    "b": {...}                # The same format as "a".
                }
        func_lib : dict
            The dict of the information of the functions.
            An example of the format of the dict elements:
                "Linear":{                   # The function name should be exactly
                                             # the same as the name of the function
                                             # variable it refers to.
                    "x_name": "x",           # The active variable of the function.
                    "param_fit": ["a", "b"], # The name of the parameters that are
                                             # involved in fitting.
                    "param_add": [],         # The name of the additional parameters
                                             # necessary for the function.
                    "operation": ["+"]       # The operation expected for this
                                             # function, for consistency check.
                                             # "+": to add with other "+" components.
                                             # "*": to multiply to other "+"
                                             # components. One model can be both "+"
                                             # and "*".
        x_list : list
            The list of active variable for the model.
        par_add_dict_all : dict
            The additional parameters for all the models in input_model_dict.
        **kwargs : dict
            Additional keywords for the ModelCombiner.
        """
        modelDict = OrderedDict()
        modelNameList = input_model_dict.keys()
        for modelName in modelNameList:
            funcName = input_model_dict[modelName]["function"]
            funcInfo = func_lib[funcName]
            xName = funcInfo["x_name"]
            #-> Build up the parameter dictionaries
            parFitList = funcInfo["param_fit"]
            parAddList = funcInfo["param_add"]
            parFitDict = OrderedDict()
            parAddDict = {}
            for parName in parFitList:
                parFitDict[parName] = input_model_dict[modelName][parName]
            for parName in parAddList:
                par_add_iterm = par_add_dict_all.get(parName, "No this parameter")
                if par_add_iterm == "No this parameter":
                    pass
                else:
                    parAddDict[parName] = par_add_iterm
            #-> Check the consistency if the component is multiply
            multiList = input_model_dict[modelName].get("multiply", None)
            if not multiList is None:
                #--> The "*" should be included in the operation list.
                assert "*" in funcInfo["operation"]
                if not QuietMode:
                    print "[Model_Generator]: {0} is multiplied to {1}!".format(modelName, multiList)
                #--> Check further the target models are not multiplicative.
                for tmn in multiList:
                    f_mlt = input_model_dict[tmn].get("multiply", None)
                    if not f_mlt is None:
                        raise ValueError("The multiList includes a multiplicative model ({0})!".format(tmn))
            modelDict[modelName] = ModelFunction(funcName, xName, parFitDict, parAddDict, multiList)
        ModelCombiner.__init__(self, modelDict, x_list, np.complex_, **kwargs)

    def Visibility(self, *xpars):
        """
        Calculate the complex visibility.

        Parameters
        ----------
        u : array_like (optional)
            The u coordinates.
        v : array_like (optional)
            The v coordinates.

        Returns
        -------
        Complex visibility at the input or prior uv position.
        """
        return self.combineResult(*xpars)

    def Amplitude(self, *xpars):
        """
        Calculate the amplitude of the visibility.

        Parameters
        ----------
        u : array_like (optional)
            The u coordinates.
        v : array_like (optional)
            The v coordinates.

        Returns
        -------
        Visibility amplitude at the input or prior uv position.
        """
        return np.absolute(self.combineResult(*xpars))

    def Phase(self, *xpars):
        """
        Calculate the phase of the visibility.

        Parameters
        ----------
        u : array_like (optional)
            The u coordinates.
        v : array_like (optional)
            The v coordinates.

        Returns
        -------
        Visibility phase at the input or prior uv position.
        """
        return np.angle(self.combineResult(*xpars))

    def Closure_Phase(self, uv1, uv2, uv3):
        """
        Calculate the closure phase with the given baselines.
        Note that:
            closure phase = phi1 + phi2 - phi3

        Parameters
        ----------
        uv1 : list (u, v)
            The uv coordinates of the first baseline.
        uv2 : list (u, v)
            The uv coordinates of the second baseline.  Note that this is the
            minus term.
        uv3 : list (u, v)
            The uv coordinates of the third baseline.

        Returns
        -------
        cphi : array_like
            The closure phase defined as phi1 + phi2 - phi3.
        """
        phi1 = self.Phase(*uv1)
        phi2 = self.Phase(*uv2)
        phi3 = self.Phase(*uv3)
        cphi = phi1 - phi2 + phi3
        cphi[cphi < -pi] += 2 * pi
        cphi[cphi > pi] -= 2 * pi
        return cphi
