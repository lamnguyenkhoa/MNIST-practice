��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.02unknown8��
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�* 
shared_nameconv2d_2/kernel
|
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:@�*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:�*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�d*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:d*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d$*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:d$*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:$*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
SGD/conv2d/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/conv2d/kernel/momentum
�
.SGD/conv2d/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d/kernel/momentum*&
_output_shapes
: *
dtype0
�
SGD/conv2d/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameSGD/conv2d/bias/momentum
�
,SGD/conv2d/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d/bias/momentum*
_output_shapes
: *
dtype0
�
&SGD/batch_normalization/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&SGD/batch_normalization/gamma/momentum
�
:SGD/batch_normalization/gamma/momentum/Read/ReadVariableOpReadVariableOp&SGD/batch_normalization/gamma/momentum*
_output_shapes
: *
dtype0
�
%SGD/batch_normalization/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%SGD/batch_normalization/beta/momentum
�
9SGD/batch_normalization/beta/momentum/Read/ReadVariableOpReadVariableOp%SGD/batch_normalization/beta/momentum*
_output_shapes
: *
dtype0
�
SGD/conv2d_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*-
shared_nameSGD/conv2d_1/kernel/momentum
�
0SGD/conv2d_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1/kernel/momentum*&
_output_shapes
: @*
dtype0
�
SGD/conv2d_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/conv2d_1/bias/momentum
�
.SGD/conv2d_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1/bias/momentum*
_output_shapes
:@*
dtype0
�
(SGD/batch_normalization_1/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(SGD/batch_normalization_1/gamma/momentum
�
<SGD/batch_normalization_1/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_1/gamma/momentum*
_output_shapes
:@*
dtype0
�
'SGD/batch_normalization_1/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'SGD/batch_normalization_1/beta/momentum
�
;SGD/batch_normalization_1/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_1/beta/momentum*
_output_shapes
:@*
dtype0
�
SGD/conv2d_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*-
shared_nameSGD/conv2d_2/kernel/momentum
�
0SGD/conv2d_2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_2/kernel/momentum*'
_output_shapes
:@�*
dtype0
�
SGD/conv2d_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameSGD/conv2d_2/bias/momentum
�
.SGD/conv2d_2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_2/bias/momentum*
_output_shapes	
:�*
dtype0
�
(SGD/batch_normalization_2/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(SGD/batch_normalization_2/gamma/momentum
�
<SGD/batch_normalization_2/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_2/gamma/momentum*
_output_shapes	
:�*
dtype0
�
'SGD/batch_normalization_2/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'SGD/batch_normalization_2/beta/momentum
�
;SGD/batch_normalization_2/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_2/beta/momentum*
_output_shapes	
:�*
dtype0
�
SGD/dense/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d**
shared_nameSGD/dense/kernel/momentum
�
-SGD/dense/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/kernel/momentum*
_output_shapes
:	�d*
dtype0
�
SGD/dense/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*(
shared_nameSGD/dense/bias/momentum

+SGD/dense/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/bias/momentum*
_output_shapes
:d*
dtype0
�
SGD/dense_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d$*,
shared_nameSGD/dense_1/kernel/momentum
�
/SGD/dense_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_1/kernel/momentum*
_output_shapes

:d$*
dtype0
�
SGD/dense_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameSGD/dense_1/bias/momentum
�
-SGD/dense_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_1/bias/momentum*
_output_shapes
:$*
dtype0

NoOpNoOp
�U
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�T
value�TB�T B�T
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
�
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
	variables
 trainable_variables
!	keras_api
R
"regularization_losses
#	variables
$trainable_variables
%	keras_api
h

&kernel
'bias
(regularization_losses
)	variables
*trainable_variables
+	keras_api
�
,axis
	-gamma
.beta
/moving_mean
0moving_variance
1regularization_losses
2	variables
3trainable_variables
4	keras_api
h

5kernel
6bias
7regularization_losses
8	variables
9trainable_variables
:	keras_api
�
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
R
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
R
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
h

Lkernel
Mbias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
R
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
h

Vkernel
Wbias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
�
\iter
	]decay
^learning_rate
_momentummomentum�momentum�momentum�momentum�&momentum�'momentum�-momentum�.momentum�5momentum�6momentum�<momentum�=momentum�Lmomentum�Mmomentum�Vmomentum�Wmomentum�
 
�
0
1
2
3
4
5
&6
'7
-8
.9
/10
011
512
613
<14
=15
>16
?17
L18
M19
V20
W21
v
0
1
2
3
&4
'5
-6
.7
58
69
<10
=11
L12
M13
V14
W15
�
`layer_metrics
anon_trainable_variables
regularization_losses

blayers
clayer_regularization_losses
	variables
trainable_variables
dmetrics
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
elayer_metrics
fnon_trainable_variables
regularization_losses

glayers
hlayer_regularization_losses
	variables
trainable_variables
imetrics
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

0
1
�
jlayer_metrics
knon_trainable_variables
regularization_losses

llayers
mlayer_regularization_losses
	variables
 trainable_variables
nmetrics
 
 
 
�
olayer_metrics
pnon_trainable_variables
"regularization_losses

qlayers
rlayer_regularization_losses
#	variables
$trainable_variables
smetrics
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
�
tlayer_metrics
unon_trainable_variables
(regularization_losses

vlayers
wlayer_regularization_losses
)	variables
*trainable_variables
xmetrics
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1
/2
03

-0
.1
�
ylayer_metrics
znon_trainable_variables
1regularization_losses

{layers
|layer_regularization_losses
2	variables
3trainable_variables
}metrics
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61

50
61
�
~layer_metrics
non_trainable_variables
7regularization_losses
�layers
 �layer_regularization_losses
8	variables
9trainable_variables
�metrics
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1
>2
?3

<0
=1
�
�layer_metrics
�non_trainable_variables
@regularization_losses
�layers
 �layer_regularization_losses
A	variables
Btrainable_variables
�metrics
 
 
 
�
�layer_metrics
�non_trainable_variables
Dregularization_losses
�layers
 �layer_regularization_losses
E	variables
Ftrainable_variables
�metrics
 
 
 
�
�layer_metrics
�non_trainable_variables
Hregularization_losses
�layers
 �layer_regularization_losses
I	variables
Jtrainable_variables
�metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
�
�layer_metrics
�non_trainable_variables
Nregularization_losses
�layers
 �layer_regularization_losses
O	variables
Ptrainable_variables
�metrics
 
 
 
�
�layer_metrics
�non_trainable_variables
Rregularization_losses
�layers
 �layer_regularization_losses
S	variables
Ttrainable_variables
�metrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1

V0
W1
�
�layer_metrics
�non_trainable_variables
Xregularization_losses
�layers
 �layer_regularization_losses
Y	variables
Ztrainable_variables
�metrics
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
*
0
1
/2
03
>4
?5
V
0
1
2
3
4
5
6
7
	8

9
10
11
 

�0
�1
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

/0
01
 
 
 
 
 
 
 
 
 

>0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
��
VARIABLE_VALUESGD/conv2d/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv2d/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE&SGD/batch_normalization/gamma/momentumXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%SGD/batch_normalization/beta/momentumWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv2d_1/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv2d_1/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(SGD/batch_normalization_1/gamma/momentumXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'SGD/batch_normalization_1/beta/momentumWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv2d_2/kernel/momentumYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv2d_2/bias/momentumWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(SGD/batch_normalization_2/gamma/momentumXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'SGD/batch_normalization_2/beta/momentumWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense/kernel/momentumYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense/bias/momentumWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense_1/kernel/momentumYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense_1/bias/momentumWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_conv2d_inputPlaceholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_111973
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.SGD/conv2d/kernel/momentum/Read/ReadVariableOp,SGD/conv2d/bias/momentum/Read/ReadVariableOp:SGD/batch_normalization/gamma/momentum/Read/ReadVariableOp9SGD/batch_normalization/beta/momentum/Read/ReadVariableOp0SGD/conv2d_1/kernel/momentum/Read/ReadVariableOp.SGD/conv2d_1/bias/momentum/Read/ReadVariableOp<SGD/batch_normalization_1/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_1/beta/momentum/Read/ReadVariableOp0SGD/conv2d_2/kernel/momentum/Read/ReadVariableOp.SGD/conv2d_2/bias/momentum/Read/ReadVariableOp<SGD/batch_normalization_2/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_2/beta/momentum/Read/ReadVariableOp-SGD/dense/kernel/momentum/Read/ReadVariableOp+SGD/dense/bias/momentum/Read/ReadVariableOp/SGD/dense_1/kernel/momentum/Read/ReadVariableOp-SGD/dense_1/bias/momentum/Read/ReadVariableOpConst*;
Tin4
220	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_112939
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1SGD/conv2d/kernel/momentumSGD/conv2d/bias/momentum&SGD/batch_normalization/gamma/momentum%SGD/batch_normalization/beta/momentumSGD/conv2d_1/kernel/momentumSGD/conv2d_1/bias/momentum(SGD/batch_normalization_1/gamma/momentum'SGD/batch_normalization_1/beta/momentumSGD/conv2d_2/kernel/momentumSGD/conv2d_2/bias/momentum(SGD/batch_normalization_2/gamma/momentum'SGD/batch_normalization_2/beta/momentumSGD/dense/kernel/momentumSGD/dense/bias/momentumSGD/dense_1/kernel/momentumSGD/dense_1/bias/momentum*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_113087��
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_111178

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
}
(__inference_dense_1_layer_call_fn_112778

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1116202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_111620

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������$2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_111807
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1117602
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameconv2d_input
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_111209

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_112207

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1117602
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_112722

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
D
(__inference_flatten_layer_call_fn_112711

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1115442
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_110989

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� :::::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_layer_call_fn_112391

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1109582
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�>
�
F__inference_sequential_layer_call_and_return_conditional_losses_111760

inputs
conv2d_111703
conv2d_111705
batch_normalization_111708
batch_normalization_111710
batch_normalization_111712
batch_normalization_111714
conv2d_1_111718
conv2d_1_111720 
batch_normalization_1_111723 
batch_normalization_1_111725 
batch_normalization_1_111727 
batch_normalization_1_111729
conv2d_2_111732
conv2d_2_111734 
batch_normalization_2_111737 
batch_normalization_2_111739 
batch_normalization_2_111741 
batch_normalization_2_111743
dense_111748
dense_111750
dense_1_111754
dense_1_111756
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_111703conv2d_111705*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1112472 
conv2d/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_111708batch_normalization_111710batch_normalization_111712batch_normalization_111714*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1112822-
+batch_normalization/StatefulPartitionedCall�
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1110062
max_pooling2d/PartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_111718conv2d_1_111720*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1113482"
 conv2d_1/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_111723batch_normalization_1_111725batch_normalization_1_111727batch_normalization_1_111729*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1113832/
-batch_normalization_1/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_111732conv2d_2_111734*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1114482"
 conv2d_2/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_111737batch_normalization_2_111739batch_normalization_2_111741batch_normalization_2_111743*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1114832/
-batch_normalization_2/StatefulPartitionedCall�
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1112262!
max_pooling2d_1/PartitionedCall�
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1115442
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_111748dense_111750*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1115632
dense/StatefulPartitionedCall�
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1115912!
dropout/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_111754dense_1_111756*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1116202!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_112296

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
B__inference_conv2d_layer_call_and_return_conditional_losses_112267

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112526

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@:::::W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
D
(__inference_dropout_layer_call_fn_112758

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1115962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
B__inference_conv2d_layer_call_and_return_conditional_losses_111247

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_2_layer_call_fn_112687

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1114832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:���������		�::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�	
�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_111448

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@:::W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�u
�
F__inference_sequential_layer_call_and_return_conditional_losses_112072

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�$batch_normalization_2/AssignNewValue�&batch_normalization_2/AssignNewValue_1�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv2d/Conv2D�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d/Relu�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2&
$batch_normalization/FusedBatchNormV3�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1�
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
conv2d_1/Conv2D�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_1/Relu�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_1/FusedBatchNormV3�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue�
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingVALID*
strides
2
conv2d_2/Conv2D�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
conv2d_2/Relu�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������		�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_2/FusedBatchNormV3�
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue�
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1�
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_2/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten/Const�
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/dropout/Const�
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������d2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:���������d*
dtype02.
,dropout/dropout/random_uniform/RandomUniform�
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2 
dropout/dropout/GreaterEqual/y�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:���������d2
dropout/dropout/Mul_1�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:d$*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������$2
dense_1/Softmax�
IdentityIdentitydense_1/Softmax:softmax:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_layer_call_fn_112327

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1112822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
{
&__inference_dense_layer_call_fn_112731

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1115632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_112563

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@:::W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_110958

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_111973
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_1108962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameconv2d_input
�
J
.__inference_max_pooling2d_layer_call_fn_111012

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1110062
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112462

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@:::::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_112360

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112508

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_112743

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_2_layer_call_fn_112700

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1115012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:���������		�::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_111105

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@:::::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�=
�
F__inference_sequential_layer_call_and_return_conditional_losses_111697
conv2d_input
conv2d_111640
conv2d_111642
batch_normalization_111645
batch_normalization_111647
batch_normalization_111649
batch_normalization_111651
conv2d_1_111655
conv2d_1_111657 
batch_normalization_1_111660 
batch_normalization_1_111662 
batch_normalization_1_111664 
batch_normalization_1_111666
conv2d_2_111669
conv2d_2_111671 
batch_normalization_2_111674 
batch_normalization_2_111676 
batch_normalization_2_111678 
batch_normalization_2_111680
dense_111685
dense_111687
dense_1_111691
dense_1_111693
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_111640conv2d_111642*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1112472 
conv2d/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_111645batch_normalization_111647batch_normalization_111649batch_normalization_111651*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1113002-
+batch_normalization/StatefulPartitionedCall�
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1110062
max_pooling2d/PartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_111655conv2d_1_111657*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1113482"
 conv2d_1/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_111660batch_normalization_1_111662batch_normalization_1_111664batch_normalization_1_111666*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1114012/
-batch_normalization_1/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_111669conv2d_2_111671*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1114482"
 conv2d_2/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_111674batch_normalization_2_111676batch_normalization_2_111678batch_normalization_2_111680*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1115012/
-batch_normalization_2/StatefulPartitionedCall�
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1112262!
max_pooling2d_1/PartitionedCall�
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1115442
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_111685dense_111687*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1115632
dense/StatefulPartitionedCall�
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1115962
dropout/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_111691dense_1_111693*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1116202!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameconv2d_input
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_111401

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@:::::W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_111563

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_111348

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:��������� :::W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�>
�
F__inference_sequential_layer_call_and_return_conditional_losses_111637
conv2d_input
conv2d_111258
conv2d_111260
batch_normalization_111327
batch_normalization_111329
batch_normalization_111331
batch_normalization_111333
conv2d_1_111359
conv2d_1_111361 
batch_normalization_1_111428 
batch_normalization_1_111430 
batch_normalization_1_111432 
batch_normalization_1_111434
conv2d_2_111459
conv2d_2_111461 
batch_normalization_2_111528 
batch_normalization_2_111530 
batch_normalization_2_111532 
batch_normalization_2_111534
dense_111574
dense_111576
dense_1_111631
dense_1_111633
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_111258conv2d_111260*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1112472 
conv2d/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_111327batch_normalization_111329batch_normalization_111331batch_normalization_111333*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1112822-
+batch_normalization/StatefulPartitionedCall�
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1110062
max_pooling2d/PartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_111359conv2d_1_111361*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1113482"
 conv2d_1/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_111428batch_normalization_1_111430batch_normalization_1_111432batch_normalization_1_111434*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1113832/
-batch_normalization_1/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_111459conv2d_2_111461*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1114482"
 conv2d_2/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_111528batch_normalization_2_111530batch_normalization_2_111532batch_normalization_2_111534*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1114832/
-batch_normalization_2/StatefulPartitionedCall�
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1112262!
max_pooling2d_1/PartitionedCall�
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1115442
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_111574dense_111576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1115632
dense/StatefulPartitionedCall�
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1115912!
dropout/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_111631dense_1_111633*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1116202!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameconv2d_input
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_112748

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112656

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������		�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:���������		�::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_layer_call_fn_112404

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1109892
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_111006

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_112706

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_111544

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_112256

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1118692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
~
)__inference_conv2d_2_layer_call_fn_112572

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1114482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�b
�
__inference__traced_save_112939
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_sgd_conv2d_kernel_momentum_read_readvariableop7
3savev2_sgd_conv2d_bias_momentum_read_readvariableopE
Asavev2_sgd_batch_normalization_gamma_momentum_read_readvariableopD
@savev2_sgd_batch_normalization_beta_momentum_read_readvariableop;
7savev2_sgd_conv2d_1_kernel_momentum_read_readvariableop9
5savev2_sgd_conv2d_1_bias_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_1_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_1_beta_momentum_read_readvariableop;
7savev2_sgd_conv2d_2_kernel_momentum_read_readvariableop9
5savev2_sgd_conv2d_2_bias_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_2_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_2_beta_momentum_read_readvariableop8
4savev2_sgd_dense_kernel_momentum_read_readvariableop6
2savev2_sgd_dense_bias_momentum_read_readvariableop:
6savev2_sgd_dense_1_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_1_bias_momentum_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ee575da99f2a4443a08d8ae155d268aa/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*�
value�B�/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_sgd_conv2d_kernel_momentum_read_readvariableop3savev2_sgd_conv2d_bias_momentum_read_readvariableopAsavev2_sgd_batch_normalization_gamma_momentum_read_readvariableop@savev2_sgd_batch_normalization_beta_momentum_read_readvariableop7savev2_sgd_conv2d_1_kernel_momentum_read_readvariableop5savev2_sgd_conv2d_1_bias_momentum_read_readvariableopCsavev2_sgd_batch_normalization_1_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_1_beta_momentum_read_readvariableop7savev2_sgd_conv2d_2_kernel_momentum_read_readvariableop5savev2_sgd_conv2d_2_bias_momentum_read_readvariableopCsavev2_sgd_batch_normalization_2_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_2_beta_momentum_read_readvariableop4savev2_sgd_dense_kernel_momentum_read_readvariableop2savev2_sgd_dense_bias_momentum_read_readvariableop6savev2_sgd_dense_1_kernel_momentum_read_readvariableop4savev2_sgd_dense_1_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *=
dtypes3
12/	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : : : @:@:@:@:@:@:@�:�:�:�:�:�:	�d:d:d$:$: : : : : : : : : : : : : @:@:@:@:@�:�:�:�:	�d:d:d$:$: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�d: 

_output_shapes
:d:$ 

_output_shapes

:d$: 

_output_shapes
:$:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: :,#(
&
_output_shapes
: @: $

_output_shapes
:@: %

_output_shapes
:@: &

_output_shapes
:@:-')
'
_output_shapes
:@�:!(

_output_shapes	
:�:!)

_output_shapes	
:�:!*

_output_shapes	
:�:%+!

_output_shapes
:	�d: ,

_output_shapes
:d:$- 

_output_shapes

:d$: .

_output_shapes
:$:/

_output_shapes
: 
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_111483

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������		�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:���������		�::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_2_layer_call_fn_112623

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1111782
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
L
0__inference_max_pooling2d_1_layer_call_fn_111232

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1112262
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_111501

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������		�:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:���������		�:::::X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112444

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
|
'__inference_conv2d_layer_call_fn_112276

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1112472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�<
�
F__inference_sequential_layer_call_and_return_conditional_losses_111869

inputs
conv2d_111812
conv2d_111814
batch_normalization_111817
batch_normalization_111819
batch_normalization_111821
batch_normalization_111823
conv2d_1_111827
conv2d_1_111829 
batch_normalization_1_111832 
batch_normalization_1_111834 
batch_normalization_1_111836 
batch_normalization_1_111838
conv2d_2_111841
conv2d_2_111843 
batch_normalization_2_111846 
batch_normalization_2_111848 
batch_normalization_2_111850 
batch_normalization_2_111852
dense_111857
dense_111859
dense_1_111863
dense_1_111865
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_111812conv2d_111814*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1112472 
conv2d/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_111817batch_normalization_111819batch_normalization_111821batch_normalization_111823*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1113002-
+batch_normalization/StatefulPartitionedCall�
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1110062
max_pooling2d/PartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_111827conv2d_1_111829*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1113482"
 conv2d_1/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_111832batch_normalization_1_111834batch_normalization_1_111836batch_normalization_1_111838*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1114012/
-batch_normalization_1/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_111841conv2d_2_111843*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1114482"
 conv2d_2/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_111846batch_normalization_2_111848batch_normalization_2_111850batch_normalization_2_111852*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1115012/
-batch_normalization_2/StatefulPartitionedCall�
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1112262!
max_pooling2d_1/PartitionedCall�
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1115442
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_111857dense_111859*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1115632
dense/StatefulPartitionedCall�
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1115962
dropout/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_111863dense_1_111865*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1116202!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�R
�	
F__inference_sequential_layer_call_and_return_conditional_losses_112158

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv2d/Conv2D�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d/Relu�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 2&
$batch_normalization/FusedBatchNormV3�
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
conv2d_1/Conv2D�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_1/Relu�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingVALID*
strides
2
conv2d_2/Conv2D�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
conv2d_2/Relu�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������		�:�:�:�:�:*
epsilon%o�:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3�
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_2/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten/Const�
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2

dense/Relu|
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:���������d2
dropout/Identity�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:d$*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������$2
dense_1/Softmaxm
IdentityIdentitydense_1/Softmax:softmax:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������:::::::::::::::::::::::W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
(__inference_dropout_layer_call_fn_112753

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1115912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�a
�
!__inference__wrapped_model_110896
conv2d_input4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource:
6sequential_batch_normalization_readvariableop_resource<
8sequential_batch_normalization_readvariableop_1_resourceK
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceM
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource<
8sequential_batch_normalization_1_readvariableop_resource>
:sequential_batch_normalization_1_readvariableop_1_resourceM
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource<
8sequential_batch_normalization_2_readvariableop_resource>
:sequential_batch_normalization_2_readvariableop_1_resourceM
Isequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identity��
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp�
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
sequential/conv2d/Conv2D�
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp�
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
sequential/conv2d/BiasAdd�
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
sequential/conv2d/Relu�
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential/batch_normalization/ReadVariableOp�
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype021
/sequential/batch_normalization/ReadVariableOp_1�
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp�
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3$sequential/conv2d/Relu:activations:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3�
 sequential/max_pooling2d/MaxPoolMaxPool3sequential/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPool�
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp�
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D�
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp�
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
sequential/conv2d_1/BiasAdd�
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
sequential/conv2d_1/Relu�
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential/batch_normalization_1/ReadVariableOp�
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1sequential/batch_normalization_1/ReadVariableOp_1�
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02B
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&sequential/conv2d_1/Relu:activations:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 23
1sequential/batch_normalization_1/FusedBatchNormV3�
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp�
sequential/conv2d_2/Conv2DConv2D5sequential/batch_normalization_1/FusedBatchNormV3:y:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingVALID*
strides
2
sequential/conv2d_2/Conv2D�
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp�
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2
sequential/conv2d_2/BiasAdd�
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
sequential/conv2d_2/Relu�
/sequential/batch_normalization_2/ReadVariableOpReadVariableOp8sequential_batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype021
/sequential/batch_normalization_2/ReadVariableOp�
1sequential/batch_normalization_2/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype023
1sequential/batch_normalization_2/ReadVariableOp_1�
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02D
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
1sequential/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3&sequential/conv2d_2/Relu:activations:07sequential/batch_normalization_2/ReadVariableOp:value:09sequential/batch_normalization_2/ReadVariableOp_1:value:0Hsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������		�:�:�:�:�:*
epsilon%o�:*
is_training( 23
1sequential/batch_normalization_2/FusedBatchNormV3�
"sequential/max_pooling2d_1/MaxPoolMaxPool5sequential/batch_normalization_2/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPool�
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
sequential/flatten/Const�
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_1/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:����������2
sequential/flatten/Reshape�
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02(
&sequential/dense/MatMul/ReadVariableOp�
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
sequential/dense/MatMul�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
sequential/dense/BiasAdd�
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
sequential/dense/Relu�
sequential/dropout/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*'
_output_shapes
:���������d2
sequential/dropout/Identity�
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:d$*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp�
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
sequential/dense_1/MatMul�
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp�
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
sequential/dense_1/BiasAdd�
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������$2
sequential/dense_1/Softmaxx
IdentityIdentity$sequential/dense_1/Softmax:softmax:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������:::::::::::::::::::::::] Y
/
_output_shapes
:���������
&
_user_specified_nameconv2d_input
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_112769

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������$2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_1_layer_call_fn_112539

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1113832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_1_layer_call_fn_112488

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1111052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_112314

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� :::::W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_111074

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
~
)__inference_conv2d_1_layer_call_fn_112424

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1113482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_1_layer_call_fn_112552

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1114012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_2_layer_call_fn_112636

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1112092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_111591

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_112378

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� :::::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112674

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������		�:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:���������		�:::::X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_111596

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112610

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_layer_call_fn_112340

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1113002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_111383

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_111282

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_112415

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:��������� :::W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_111916
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1118692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameconv2d_input
�
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_111226

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_1_layer_call_fn_112475

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1110742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_111300

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� :::::W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112592

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_113087
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias0
,assignvariableop_2_batch_normalization_gamma/
+assignvariableop_3_batch_normalization_beta6
2assignvariableop_4_batch_normalization_moving_mean:
6assignvariableop_5_batch_normalization_moving_variance&
"assignvariableop_6_conv2d_1_kernel$
 assignvariableop_7_conv2d_1_bias2
.assignvariableop_8_batch_normalization_1_gamma1
-assignvariableop_9_batch_normalization_1_beta9
5assignvariableop_10_batch_normalization_1_moving_mean=
9assignvariableop_11_batch_normalization_1_moving_variance'
#assignvariableop_12_conv2d_2_kernel%
!assignvariableop_13_conv2d_2_bias3
/assignvariableop_14_batch_normalization_2_gamma2
.assignvariableop_15_batch_normalization_2_beta9
5assignvariableop_16_batch_normalization_2_moving_mean=
9assignvariableop_17_batch_normalization_2_moving_variance$
 assignvariableop_18_dense_kernel"
assignvariableop_19_dense_bias&
"assignvariableop_20_dense_1_kernel$
 assignvariableop_21_dense_1_bias 
assignvariableop_22_sgd_iter!
assignvariableop_23_sgd_decay)
%assignvariableop_24_sgd_learning_rate$
 assignvariableop_25_sgd_momentum
assignvariableop_26_total
assignvariableop_27_count
assignvariableop_28_total_1
assignvariableop_29_count_12
.assignvariableop_30_sgd_conv2d_kernel_momentum0
,assignvariableop_31_sgd_conv2d_bias_momentum>
:assignvariableop_32_sgd_batch_normalization_gamma_momentum=
9assignvariableop_33_sgd_batch_normalization_beta_momentum4
0assignvariableop_34_sgd_conv2d_1_kernel_momentum2
.assignvariableop_35_sgd_conv2d_1_bias_momentum@
<assignvariableop_36_sgd_batch_normalization_1_gamma_momentum?
;assignvariableop_37_sgd_batch_normalization_1_beta_momentum4
0assignvariableop_38_sgd_conv2d_2_kernel_momentum2
.assignvariableop_39_sgd_conv2d_2_bias_momentum@
<assignvariableop_40_sgd_batch_normalization_2_gamma_momentum?
;assignvariableop_41_sgd_batch_normalization_2_beta_momentum1
-assignvariableop_42_sgd_dense_kernel_momentum/
+assignvariableop_43_sgd_dense_bias_momentum3
/assignvariableop_44_sgd_dense_1_kernel_momentum1
-assignvariableop_45_sgd_dense_1_bias_momentum
identity_47��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*�
value�B�/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_dense_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_1_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_1_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpassignvariableop_22_sgd_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_sgd_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp%assignvariableop_24_sgd_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp assignvariableop_25_sgd_momentumIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOpassignvariableop_26_totalIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpassignvariableop_27_countIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpassignvariableop_28_total_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOpassignvariableop_29_count_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp.assignvariableop_30_sgd_conv2d_kernel_momentumIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp,assignvariableop_31_sgd_conv2d_bias_momentumIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp:assignvariableop_32_sgd_batch_normalization_gamma_momentumIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp9assignvariableop_33_sgd_batch_normalization_beta_momentumIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp0assignvariableop_34_sgd_conv2d_1_kernel_momentumIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp.assignvariableop_35_sgd_conv2d_1_bias_momentumIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp<assignvariableop_36_sgd_batch_normalization_1_gamma_momentumIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp;assignvariableop_37_sgd_batch_normalization_1_beta_momentumIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp0assignvariableop_38_sgd_conv2d_2_kernel_momentumIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp.assignvariableop_39_sgd_conv2d_2_bias_momentumIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp<assignvariableop_40_sgd_batch_normalization_2_gamma_momentumIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp;assignvariableop_41_sgd_batch_normalization_2_beta_momentumIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp-assignvariableop_42_sgd_dense_kernel_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp+assignvariableop_43_sgd_dense_bias_momentumIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp/assignvariableop_44_sgd_dense_1_kernel_momentumIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp-assignvariableop_45_sgd_dense_1_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_459
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_46�
Identity_47IdentityIdentity_46:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_47"#
identity_47Identity_47:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
M
conv2d_input=
serving_default_conv2d_input:0���������;
dense_10
StatefulPartitionedCall:0���������$tensorflow/serving/predict:Ϭ
�d
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"�_
_tf_keras_sequential�_{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
�


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
�	
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
	variables
 trainable_variables
!	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 26, 32]}}
�
"regularization_losses
#	variables
$trainable_variables
%	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

&kernel
'bias
(regularization_losses
)	variables
*trainable_variables
+	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 32]}}
�	
,axis
	-gamma
.beta
/moving_mean
0moving_variance
1regularization_losses
2	variables
3trainable_variables
4	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 64]}}
�	

5kernel
6bias
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 64]}}
�	
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 128]}}
�
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

Lkernel
Mbias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2048]}}
�
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
�

Vkernel
Wbias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
�
\iter
	]decay
^learning_rate
_momentummomentum�momentum�momentum�momentum�&momentum�'momentum�-momentum�.momentum�5momentum�6momentum�<momentum�=momentum�Lmomentum�Mmomentum�Vmomentum�Wmomentum�"
	optimizer
 "
trackable_list_wrapper
�
0
1
2
3
4
5
&6
'7
-8
.9
/10
011
512
613
<14
=15
>16
?17
L18
M19
V20
W21"
trackable_list_wrapper
�
0
1
2
3
&4
'5
-6
.7
58
69
<10
=11
L12
M13
V14
W15"
trackable_list_wrapper
�
`layer_metrics
anon_trainable_variables
regularization_losses

blayers
clayer_regularization_losses
	variables
trainable_variables
dmetrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
':% 2conv2d/kernel
: 2conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
elayer_metrics
fnon_trainable_variables
regularization_losses

glayers
hlayer_regularization_losses
	variables
trainable_variables
imetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
jlayer_metrics
knon_trainable_variables
regularization_losses

llayers
mlayer_regularization_losses
	variables
 trainable_variables
nmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
olayer_metrics
pnon_trainable_variables
"regularization_losses

qlayers
rlayer_regularization_losses
#	variables
$trainable_variables
smetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_1/kernel
:@2conv2d_1/bias
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
�
tlayer_metrics
unon_trainable_variables
(regularization_losses

vlayers
wlayer_regularization_losses
)	variables
*trainable_variables
xmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
<
-0
.1
/2
03"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
�
ylayer_metrics
znon_trainable_variables
1regularization_losses

{layers
|layer_regularization_losses
2	variables
3trainable_variables
}metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@�2conv2d_2/kernel
:�2conv2d_2/bias
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
�
~layer_metrics
non_trainable_variables
7regularization_losses
�layers
 �layer_regularization_losses
8	variables
9trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(�2batch_normalization_2/gamma
):'�2batch_normalization_2/beta
2:0� (2!batch_normalization_2/moving_mean
6:4� (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
@regularization_losses
�layers
 �layer_regularization_losses
A	variables
Btrainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
Dregularization_losses
�layers
 �layer_regularization_losses
E	variables
Ftrainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
Hregularization_losses
�layers
 �layer_regularization_losses
I	variables
Jtrainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	�d2dense/kernel
:d2
dense/bias
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
Nregularization_losses
�layers
 �layer_regularization_losses
O	variables
Ptrainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
Rregularization_losses
�layers
 �layer_regularization_losses
S	variables
Ttrainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :d$2dense_1/kernel
:$2dense_1/bias
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
Xregularization_losses
�layers
 �layer_regularization_losses
Y	variables
Ztrainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_dict_wrapper
J
0
1
/2
03
>4
?5"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
2:0 2SGD/conv2d/kernel/momentum
$:" 2SGD/conv2d/bias/momentum
2:0 2&SGD/batch_normalization/gamma/momentum
1:/ 2%SGD/batch_normalization/beta/momentum
4:2 @2SGD/conv2d_1/kernel/momentum
&:$@2SGD/conv2d_1/bias/momentum
4:2@2(SGD/batch_normalization_1/gamma/momentum
3:1@2'SGD/batch_normalization_1/beta/momentum
5:3@�2SGD/conv2d_2/kernel/momentum
':%�2SGD/conv2d_2/bias/momentum
5:3�2(SGD/batch_normalization_2/gamma/momentum
4:2�2'SGD/batch_normalization_2/beta/momentum
*:(	�d2SGD/dense/kernel/momentum
#:!d2SGD/dense/bias/momentum
+:)d$2SGD/dense_1/kernel/momentum
%:#$2SGD/dense_1/bias/momentum
�2�
F__inference_sequential_layer_call_and_return_conditional_losses_112072
F__inference_sequential_layer_call_and_return_conditional_losses_112158
F__inference_sequential_layer_call_and_return_conditional_losses_111637
F__inference_sequential_layer_call_and_return_conditional_losses_111697�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_110896�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+
conv2d_input���������
�2�
+__inference_sequential_layer_call_fn_112256
+__inference_sequential_layer_call_fn_111807
+__inference_sequential_layer_call_fn_112207
+__inference_sequential_layer_call_fn_111916�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_conv2d_layer_call_and_return_conditional_losses_112267�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv2d_layer_call_fn_112276�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_112314
O__inference_batch_normalization_layer_call_and_return_conditional_losses_112360
O__inference_batch_normalization_layer_call_and_return_conditional_losses_112378
O__inference_batch_normalization_layer_call_and_return_conditional_losses_112296�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
4__inference_batch_normalization_layer_call_fn_112391
4__inference_batch_normalization_layer_call_fn_112327
4__inference_batch_normalization_layer_call_fn_112340
4__inference_batch_normalization_layer_call_fn_112404�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_111006�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
.__inference_max_pooling2d_layer_call_fn_111012�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_112415�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_1_layer_call_fn_112424�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112462
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112444
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112526
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112508�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_1_layer_call_fn_112488
6__inference_batch_normalization_1_layer_call_fn_112552
6__inference_batch_normalization_1_layer_call_fn_112539
6__inference_batch_normalization_1_layer_call_fn_112475�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_112563�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_2_layer_call_fn_112572�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112610
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112592
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112674
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112656�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_2_layer_call_fn_112687
6__inference_batch_normalization_2_layer_call_fn_112636
6__inference_batch_normalization_2_layer_call_fn_112700
6__inference_batch_normalization_2_layer_call_fn_112623�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_111226�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
0__inference_max_pooling2d_1_layer_call_fn_111232�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
C__inference_flatten_layer_call_and_return_conditional_losses_112706�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_flatten_layer_call_fn_112711�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dense_layer_call_and_return_conditional_losses_112722�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_layer_call_fn_112731�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dropout_layer_call_and_return_conditional_losses_112743
C__inference_dropout_layer_call_and_return_conditional_losses_112748�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_dropout_layer_call_fn_112753
(__inference_dropout_layer_call_fn_112758�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_dense_1_layer_call_and_return_conditional_losses_112769�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_1_layer_call_fn_112778�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
8B6
$__inference_signature_wrapper_111973conv2d_input�
!__inference__wrapped_model_110896�&'-./056<=>?LMVW=�:
3�0
.�+
conv2d_input���������
� "1�.
,
dense_1!�
dense_1���������$�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112444�-./0M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112462�-./0M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112508r-./0;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112526r-./0;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
6__inference_batch_normalization_1_layer_call_fn_112475�-./0M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
6__inference_batch_normalization_1_layer_call_fn_112488�-./0M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
6__inference_batch_normalization_1_layer_call_fn_112539e-./0;�8
1�.
(�%
inputs���������@
p
� " ����������@�
6__inference_batch_normalization_1_layer_call_fn_112552e-./0;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112592�<=>?N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112610�<=>?N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112656t<=>?<�9
2�/
)�&
inputs���������		�
p
� ".�+
$�!
0���������		�
� �
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112674t<=>?<�9
2�/
)�&
inputs���������		�
p 
� ".�+
$�!
0���������		�
� �
6__inference_batch_normalization_2_layer_call_fn_112623�<=>?N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
6__inference_batch_normalization_2_layer_call_fn_112636�<=>?N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
6__inference_batch_normalization_2_layer_call_fn_112687g<=>?<�9
2�/
)�&
inputs���������		�
p
� "!����������		��
6__inference_batch_normalization_2_layer_call_fn_112700g<=>?<�9
2�/
)�&
inputs���������		�
p 
� "!����������		��
O__inference_batch_normalization_layer_call_and_return_conditional_losses_112296r;�8
1�.
(�%
inputs��������� 
p
� "-�*
#� 
0��������� 
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_112314r;�8
1�.
(�%
inputs��������� 
p 
� "-�*
#� 
0��������� 
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_112360�M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_112378�M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
4__inference_batch_normalization_layer_call_fn_112327e;�8
1�.
(�%
inputs��������� 
p
� " ���������� �
4__inference_batch_normalization_layer_call_fn_112340e;�8
1�.
(�%
inputs��������� 
p 
� " ���������� �
4__inference_batch_normalization_layer_call_fn_112391�M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
4__inference_batch_normalization_layer_call_fn_112404�M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
D__inference_conv2d_1_layer_call_and_return_conditional_losses_112415l&'7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������@
� �
)__inference_conv2d_1_layer_call_fn_112424_&'7�4
-�*
(�%
inputs��������� 
� " ����������@�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_112563m567�4
-�*
(�%
inputs���������@
� ".�+
$�!
0���������		�
� �
)__inference_conv2d_2_layer_call_fn_112572`567�4
-�*
(�%
inputs���������@
� "!����������		��
B__inference_conv2d_layer_call_and_return_conditional_losses_112267l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0��������� 
� �
'__inference_conv2d_layer_call_fn_112276_7�4
-�*
(�%
inputs���������
� " ���������� �
C__inference_dense_1_layer_call_and_return_conditional_losses_112769\VW/�,
%�"
 �
inputs���������d
� "%�"
�
0���������$
� {
(__inference_dense_1_layer_call_fn_112778OVW/�,
%�"
 �
inputs���������d
� "����������$�
A__inference_dense_layer_call_and_return_conditional_losses_112722]LM0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� z
&__inference_dense_layer_call_fn_112731PLM0�-
&�#
!�
inputs����������
� "����������d�
C__inference_dropout_layer_call_and_return_conditional_losses_112743\3�0
)�&
 �
inputs���������d
p
� "%�"
�
0���������d
� �
C__inference_dropout_layer_call_and_return_conditional_losses_112748\3�0
)�&
 �
inputs���������d
p 
� "%�"
�
0���������d
� {
(__inference_dropout_layer_call_fn_112753O3�0
)�&
 �
inputs���������d
p
� "����������d{
(__inference_dropout_layer_call_fn_112758O3�0
)�&
 �
inputs���������d
p 
� "����������d�
C__inference_flatten_layer_call_and_return_conditional_losses_112706b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
(__inference_flatten_layer_call_fn_112711U8�5
.�+
)�&
inputs����������
� "������������
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_111226�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
0__inference_max_pooling2d_1_layer_call_fn_111232�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_111006�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
.__inference_max_pooling2d_layer_call_fn_111012�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
F__inference_sequential_layer_call_and_return_conditional_losses_111637�&'-./056<=>?LMVWE�B
;�8
.�+
conv2d_input���������
p

 
� "%�"
�
0���������$
� �
F__inference_sequential_layer_call_and_return_conditional_losses_111697�&'-./056<=>?LMVWE�B
;�8
.�+
conv2d_input���������
p 

 
� "%�"
�
0���������$
� �
F__inference_sequential_layer_call_and_return_conditional_losses_112072�&'-./056<=>?LMVW?�<
5�2
(�%
inputs���������
p

 
� "%�"
�
0���������$
� �
F__inference_sequential_layer_call_and_return_conditional_losses_112158�&'-./056<=>?LMVW?�<
5�2
(�%
inputs���������
p 

 
� "%�"
�
0���������$
� �
+__inference_sequential_layer_call_fn_111807y&'-./056<=>?LMVWE�B
;�8
.�+
conv2d_input���������
p

 
� "����������$�
+__inference_sequential_layer_call_fn_111916y&'-./056<=>?LMVWE�B
;�8
.�+
conv2d_input���������
p 

 
� "����������$�
+__inference_sequential_layer_call_fn_112207s&'-./056<=>?LMVW?�<
5�2
(�%
inputs���������
p

 
� "����������$�
+__inference_sequential_layer_call_fn_112256s&'-./056<=>?LMVW?�<
5�2
(�%
inputs���������
p 

 
� "����������$�
$__inference_signature_wrapper_111973�&'-./056<=>?LMVWM�J
� 
C�@
>
conv2d_input.�+
conv2d_input���������"1�.
,
dense_1!�
dense_1���������$