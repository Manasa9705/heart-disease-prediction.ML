{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "bbd660b62cc616efa33651beedc49b86045b784d"
   },
   "source": [
    "## <font size=5> <strong>Heart Disease Prediction By Shreekant Gosavi\n",
    " "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "e41ea25bec5928203cec544d0413fecd4b4e5555"
   },
   "source": [
    "## I. Importing essential libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "_uuid": "f571f7e57c828d45fe55f6136fe8c2e796f74d4e"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['.ipynb_checkpoints', 'heart.csv', 'Heart_disease_prediction.ipynb', 'README.md']\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "%matplotlib inline\n",
    "\n",
    "import os\n",
    "print(os.listdir())\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "44e71221837f6fa60edc2c83b7492ddb019cc1cd"
   },
   "source": [
    "## II. Importing and understanding our dataset "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "_uuid": "2a1a1dae64ae3c934849b2b918bc7d68cd59e3f6"
   },
   "outputs": [],
   "source": [
    "dataset = pd.read_csv(\"heart.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "481fa1d160a3256ef2470535bfb0574820fbaabd"
   },
   "source": [
    "#### Verifying it as a 'dataframe' object in pandas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "_uuid": "86353d54a331dbf55a63874402cf13e2a72c3750"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "pandas.core.frame.DataFrame"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(dataset)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "44649a50ce58d2e10a032f7d0e7ecf435e932481"
   },
   "source": [
    "#### Shape of dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "_uuid": "0a2396061d262bee451e61dd51be84d0bd1ac9d0"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(303, 14)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "3e1de0c39fc28f086a5e8377cc5fbdbf91d377b3"
   },
   "source": [
    "#### Printing out a few columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "_uuid": "87ebcc578e5959fe9a9c9a538c73122183454459"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>145</td>\n",
       "      <td>233</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>150</td>\n",
       "      <td>0</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>130</td>\n",
       "      <td>250</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>187</td>\n",
       "      <td>0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>41</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>130</td>\n",
       "      <td>204</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>172</td>\n",
       "      <td>0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>56</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>120</td>\n",
       "      <td>236</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>178</td>\n",
       "      <td>0</td>\n",
       "      <td>0.8</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>120</td>\n",
       "      <td>354</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>163</td>\n",
       "      <td>1</td>\n",
       "      <td>0.6</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  sex  cp  trestbps  chol   ...    oldpeak  slope  ca  thal  target\n",
       "0   63    1   3       145   233   ...        2.3      0   0     1       1\n",
       "1   37    1   2       130   250   ...        3.5      0   0     2       1\n",
       "2   41    0   1       130   204   ...        1.4      2   0     2       1\n",
       "3   56    1   1       120   236   ...        0.8      2   0     2       1\n",
       "4   57    0   0       120   354   ...        0.6      2   0     2       1\n",
       "\n",
       "[5 rows x 14 columns]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "_uuid": "5132eb43114bf99d5f857f459d0c9d2faffc9644"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>248</th>\n",
       "      <td>54</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>192</td>\n",
       "      <td>283</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>195</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>60</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>150</td>\n",
       "      <td>240</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>171</td>\n",
       "      <td>0</td>\n",
       "      <td>0.9</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>239</th>\n",
       "      <td>35</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>126</td>\n",
       "      <td>282</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>156</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>120</td>\n",
       "      <td>354</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>163</td>\n",
       "      <td>1</td>\n",
       "      <td>0.6</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>44</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>120</td>\n",
       "      <td>263</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>173</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     age  sex  cp  trestbps  chol   ...    oldpeak  slope  ca  thal  target\n",
       "248   54    1   1       192   283   ...        0.0      2   1     3       0\n",
       "147   60    0   3       150   240   ...        0.9      2   0     2       1\n",
       "239   35    1   0       126   282   ...        0.0      2   0     3       0\n",
       "4     57    0   0       120   354   ...        0.6      2   0     2       1\n",
       "7     44    1   1       120   263   ...        0.0      2   0     3       1\n",
       "\n",
       "[5 rows x 14 columns]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.sample(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "1113236bec2848d33c5bfe088ff0d03246b8b7ce"
   },
   "source": [
    "#### Description"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "_uuid": "c31619815cb0dae5586985671fdc21110b39a821"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>54.366337</td>\n",
       "      <td>0.683168</td>\n",
       "      <td>0.966997</td>\n",
       "      <td>131.623762</td>\n",
       "      <td>246.264026</td>\n",
       "      <td>0.148515</td>\n",
       "      <td>0.528053</td>\n",
       "      <td>149.646865</td>\n",
       "      <td>0.326733</td>\n",
       "      <td>1.039604</td>\n",
       "      <td>1.399340</td>\n",
       "      <td>0.729373</td>\n",
       "      <td>2.313531</td>\n",
       "      <td>0.544554</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>9.082101</td>\n",
       "      <td>0.466011</td>\n",
       "      <td>1.032052</td>\n",
       "      <td>17.538143</td>\n",
       "      <td>51.830751</td>\n",
       "      <td>0.356198</td>\n",
       "      <td>0.525860</td>\n",
       "      <td>22.905161</td>\n",
       "      <td>0.469794</td>\n",
       "      <td>1.161075</td>\n",
       "      <td>0.616226</td>\n",
       "      <td>1.022606</td>\n",
       "      <td>0.612277</td>\n",
       "      <td>0.498835</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>29.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>94.000000</td>\n",
       "      <td>126.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>71.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>47.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>120.000000</td>\n",
       "      <td>211.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>133.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>55.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>130.000000</td>\n",
       "      <td>240.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>153.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>61.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>140.000000</td>\n",
       "      <td>274.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>166.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.600000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>77.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>200.000000</td>\n",
       "      <td>564.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>202.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>6.200000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              age         sex     ...            thal      target\n",
       "count  303.000000  303.000000     ...      303.000000  303.000000\n",
       "mean    54.366337    0.683168     ...        2.313531    0.544554\n",
       "std      9.082101    0.466011     ...        0.612277    0.498835\n",
       "min     29.000000    0.000000     ...        0.000000    0.000000\n",
       "25%     47.500000    0.000000     ...        2.000000    0.000000\n",
       "50%     55.000000    1.000000     ...        2.000000    1.000000\n",
       "75%     61.000000    1.000000     ...        3.000000    1.000000\n",
       "max     77.000000    1.000000     ...        3.000000    1.000000\n",
       "\n",
       "[8 rows x 14 columns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "_uuid": "718b82039841c137ab7e08a6e79e264643134642"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 303 entries, 0 to 302\n",
      "Data columns (total 14 columns):\n",
      "age         303 non-null int64\n",
      "sex         303 non-null int64\n",
      "cp          303 non-null int64\n",
      "trestbps    303 non-null int64\n",
      "chol        303 non-null int64\n",
      "fbs         303 non-null int64\n",
      "restecg     303 non-null int64\n",
      "thalach     303 non-null int64\n",
      "exang       303 non-null int64\n",
      "oldpeak     303 non-null float64\n",
      "slope       303 non-null int64\n",
      "ca          303 non-null int64\n",
      "thal        303 non-null int64\n",
      "target      303 non-null int64\n",
      "dtypes: float64(1), int64(13)\n",
      "memory usage: 33.2 KB\n"
     ]
    }
   ],
   "sour
