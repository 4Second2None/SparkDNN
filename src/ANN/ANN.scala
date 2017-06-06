package ANN

/**
  * Created by root on 17-3-22.
  */
import breeze.linalg.DenseMatrix
import breeze.numerics.{exp, tanh}

import scala.collection.mutable.ArrayBuffer
import java.util.Random

import org.apache.spark.Logging
import org.apache.spark.rdd.RDD

import scala.math._

case class ANNLabel(lable: DenseMatrix[Double], nna: ArrayBuffer[DenseMatrix[Double]], error: DenseMatrix[Double]) extends Serializable

case class ANNParameter(size: Array[Int], layer: Int, active_Func: String, lastout_Func: String, alpha: Double, momentum: Double) extends Serializable
class ANN(
           private var size: Array[Int],
           private var layer: Int,
           private var active_Func: String,
           private var lastout_Func: String,
           private var alpha: Double,
           private var momentum: Double,
           private var initW: Array[DenseMatrix[Double]]
         ) extends Serializable with Logging{

  def this() = this(Array[Int](10, 5, 1), 3, "tan", "sigm", 2.0, 0.5, Array(DenseMatrix.zeros[Double](1, 1)))

  /** 设置神经网络结构. Default: [10, 5, 1]. */
  def setSize(size: Array[Int]): this.type = {
    this.size = size
    this
  }

  /** 设置神经网络层数据. Default: 3. */
  def setLayer(layer: Int): this.type = {
    this.layer = layer
    this
  }

  /** 设置隐含层函数. Default: sigm. */
  def setLastout_function(lastout_Func: String): this.type = {
    this.lastout_Func = lastout_Func
    this
  }

  def setActivation_function(activation_function: String): this.type = {
    this.active_Func = activation_function
    this
  }

  /** 设置学习率因子. Default: 2. */
  def setLearningRate(learningRate: Double): this.type = {
    this.alpha = learningRate
    this
  }

  /** 设置Momentum. Default: 0.5. */
  def setMomentum(momentum: Double): this.type = {
    this.momentum = momentum
    this
  }

  def setInitW(initW: Array[DenseMatrix[Double]]): this.type = {
    this.initW = initW
    this
  }

  def train(train_d: RDD[(DenseMatrix[Double], DenseMatrix[Double])], batchSize: Int, iterate: Int): ANNModel = {
    val sc = train_d.sparkContext

    val parameter = ANNParameter(size, layer, active_Func, lastout_Func, alpha, momentum)

    var nn_W = ANNExt.InitialWeight(size)

    if (!((initW.length == 1) && (initW(0) == (DenseMatrix.zeros[Double](1, 1))))) {
      for (i <- 0 to initW.length - 1) {
        nn_W(i) = initW(i)
      }
    }

    var nn_vW = ANNExt.InitialWeightV(size)

    val m = train_d.count

    val batchNum = (m / batchSize).toInt

    for(i <- 1 to iterate){
      val bc_parameter = sc.broadcast(parameter)

      val splitRate = Array.fill(batchNum)(1.0 / batchNum)

      for (b <- 1 to batchNum){
        val bc_nn_W = sc.broadcast(nn_W)
        val bc_nn_vW = sc.broadcast(nn_vW)

        val train_split2 = train_d.randomSplit(splitRate, System.nanoTime())
        val batch_xy1 = train_split2(b - 1)

        val train_nnff = ANNExt.ANNff(batch_xy1, bc_parameter, bc_nn_W)

        val train_nnbp = ANNExt.ANNbp(train_nnff, bc_parameter, bc_nn_W, null)

        val newWeigths = ANNExt.ANNUpdateWeights(train_nnbp, bc_parameter, bc_nn_W, bc_nn_vW)
        nn_W = newWeigths(0)
        nn_vW = newWeigths(1)
      }
    }
    new ANNModel(parameter, nn_W)
  }
}

object ANNExt extends Serializable{
  def InitialWeight(size: Array[Int]): Array[DenseMatrix[Double]] = {
    // 初始化权重参数
    // weights and weight momentum
    // nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
    // nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
    val n = size.length
    val nn_W = ArrayBuffer[DenseMatrix[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = DenseMatrix.rand(size(i), size(i - 1) + 1)
      d1 :-= 0.5
      val f1 = 2 * 4 * sqrt(6.0 / (size(i) + size(i - 1)))
      val d2 = d1 :* f1
      //val d3 = new DenseMatrix(d2.rows, d2.cols, d2.data, d2.isTranspose)
      //val d4 = Matrices.dense(d2.rows, d2.cols, d2.data)
      nn_W += d2
    }
    nn_W.toArray
  }

  /**
    * 初始化权重vW
    * 初始化为0
    */
  def InitialWeightV(size: Array[Int]): Array[DenseMatrix[Double]] = {
    // 初始化权重参数
    // weights and weight momentum
    // nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
    val n = size.length
    val nn_vW = ArrayBuffer[DenseMatrix[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = DenseMatrix.zeros[Double](size(i), size(i - 1) + 1)
      nn_vW += d1
    }
    nn_vW.toArray
  }

  /**
    * sigm激活函数
    * X = 1./(1+exp(-P));
    */
  def sigm(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    val s1 = 1.0 / (exp(matrix * (-1.0)) + 1.0)
    s1
  }

  /**
    * tanh激活函数
    * f=1.7159*tanh(2/3.*A);
    */
  def tanh_opt(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    val s1 = tanh(matrix * (2.0 / 3.0)) * 1.7159
    s1
  }

  def activeFunction(index: String, matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    val activeResult = index match{
      case "sigm" =>
        val res = sigm(matrix)
        res
      case "tan" =>
        val res = tanh_opt(matrix)
        res
      case "linear" =>
        val res = matrix
        res
    }
    activeResult
  }

  def activeDerivative(index: String, matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    val result = index match {
      case "sigm" =>
        val res = (matrix :* (1.0 - matrix))
        res
      case "tan" =>
        val res0 = (1.0 - ((matrix :* matrix) * (1.0 / (1.7159 * 1.7159))))
        val res1 = res0 * (1.7159 * (2.0 / 3.0))
        res1
    }
    result
  }


  def ANNff(
            batch_xy2: RDD[(DenseMatrix[Double], DenseMatrix[Double])],
            bc_parameter: org.apache.spark.broadcast.Broadcast[ANNParameter],
            bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[DenseMatrix[Double]]]): RDD[ANNLabel] = {

    val train_features = batch_xy2.map(b => {
      val lable = b._1
      val features = b._2
      val nna = ArrayBuffer[DenseMatrix[Double]]()
      val Bm1 = new DenseMatrix(features.rows, 1, Array.fill(features.rows * 1)(1.0))
      val featuresUpdated = DenseMatrix.horzcat(Bm1, features)
      val error = DenseMatrix.zeros[Double](lable.rows, lable.cols)

      nna += featuresUpdated

      ANNLabel(lable, nna, error)
    })

    val train_features_in_hiden = train_features.map(t => {
      val nna = t.nna

      for(h <- 1 to bc_parameter.value.layer - 2){
        val X = nna(h - 1)
        val W = bc_nn_W.value(h - 1)

        val XW = X * W.t

        val T = activeFunction(bc_parameter.value.active_Func, XW)

        val Bm1 = new DenseMatrix(T.rows, 1, Array.fill(T.rows * 1)(1.0))

        val TUpdated = DenseMatrix.horzcat(Bm1, T)

        nna += TUpdated
      }
      ANNLabel(t.lable, nna, t.error)
    })

    val train_out = train_features_in_hiden.map(t => {
      val nna = t.nna

      val lastHiden = nna(bc_parameter.value.layer - 2)
      val lastW = bc_nn_W.value(bc_parameter.value.layer - 2)

      val S = lastHiden * lastW.t

      val Y = activeFunction(bc_parameter.value.lastout_Func, S)

      nna += Y

      ANNLabel(t.lable, nna, t.error)
    })

    val train_out_with_error = train_out.map(t => {
      val lable = t.lable
      val nna = t.nna
      val predict = nna(bc_parameter.value.layer - 1)

      val error = lable - predict

      ANNLabel(t.lable, nna, error)
    })

    train_out_with_error
  }

  def ANNbp(
            train_nnff: RDD[ANNLabel],
            bc_parameter: org.apache.spark.broadcast.Broadcast[ANNParameter],
            bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[DenseMatrix[Double]]],
            bc_nn_p: org.apache.spark.broadcast.Broadcast[Array[DenseMatrix[Double]]]): Array[DenseMatrix[Double]] = {
    val lastDelta = train_nnff.map(t => {
      val nna = t.nna
      val error = t.error
      val predict = nna(bc_parameter.value.layer - 1)

      val ds = ArrayBuffer[DenseMatrix[Double]]()

      val dn = activeDerivative(bc_parameter.value.lastout_Func, predict)

      val delta = ((error * (-1.0)) :* dn)
      ds += delta

      (t, ds)
    })

    val allDelta = lastDelta.map(d => {
      val nna = d._1.nna
      val ds = d._2

      for(h <- bc_parameter.value.layer - 2 to 1 by -1){
        val hf = nna(h)
        val dn = activeDerivative(bc_parameter.value.active_Func, hf)

        val W = bc_nn_W.value(h)
        val info = if(h == bc_parameter.value.layer - 2){
          val forwardDelta = ds(bc_parameter.value.layer - 2 - h)
          val cDelta = forwardDelta * W

          val tinfo = cDelta :* dn
          tinfo
        }else{
          val forwardDelta = ds(bc_parameter.value.layer - 2 - h)(::, 1 to -1)
          val cDelta = forwardDelta * W

          val tinfo = cDelta :* dn
          tinfo
        }

        ds += info
      }

      val dw = ArrayBuffer[DenseMatrix[Double]]()

      for(h <- 0 to bc_parameter.value.layer - 2){
        val nndw = if(h == bc_parameter.value.layer - 2){
          (ds(bc_parameter.value.layer - 2 - h).t) * nna(h)
        }else{
          (ds(bc_parameter.value.layer - 2 - h)(::, 1 to -1)).t * nna(h)
        }
        dw += nndw
      }
      (d._1, dw)
    })

    val train_dw = allDelta.map((_._2))


    val initgrad = ArrayBuffer[DenseMatrix[Double]]()
    for (i <- 0 to bc_parameter.value.layer - 2) {
      val init1 = if (i + 1 == bc_parameter.value.layer - 1) {
        DenseMatrix.zeros[Double](bc_parameter.value.size(i + 1), bc_parameter.value.size(i) + 1)
      } else {
        DenseMatrix.zeros[Double](bc_parameter.value.size(i + 1), bc_parameter.value.size(i) + 1)
      }
      initgrad += init1
    }
    val (gradientSum, miniBatchSize) = train_dw.treeAggregate((initgrad, 0L))(
      seqOp = (c, v) => {
        // c: (grad, count), v: (grad)
        val grad1 = c._1
        val grad2 = v
        val sumgrad = ArrayBuffer[DenseMatrix[Double]]()
        for (i <- 0 to bc_parameter.value.layer - 2) {
          val Bm1 = grad1(i)
          val Bm2 = grad2(i)
          val Bmsum = Bm1 + Bm2
          sumgrad += Bmsum
        }
        (sumgrad, c._2 + 1)
      },
      combOp = (c1, c2) => {
        // c: (grad, count)
        val grad1 = c1._1
        val grad2 = c2._1
        val sumgrad = ArrayBuffer[DenseMatrix[Double]]()
        for (i <- 0 to bc_parameter.value.layer - 2) {
          val Bm1 = grad1(i)
          val Bm2 = grad2(i)
          val Bmsum = Bm1 + Bm2
          sumgrad += Bmsum
        }
        (sumgrad, c1._2 + c2._2)
      })

    val gradientAvg = ArrayBuffer[DenseMatrix[Double]]()
    for (i <- 0 to bc_parameter.value.layer - 2) {
      val Bm1 = gradientSum(i)
      val Bmavg = Bm1 :/ miniBatchSize.toDouble
      gradientAvg += Bmavg
    }
    gradientAvg.toArray
  }

  def ANNUpdateWeights(train_nnbp: Array[DenseMatrix[Double]],
                       bc_parameter: org.apache.spark.broadcast.Broadcast[ANNParameter],
                       bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[DenseMatrix[Double]]],
                       bc_nn_vW: org.apache.spark.broadcast.Broadcast[Array[DenseMatrix[Double]]]): Array[Array[DenseMatrix[Double]]] = {
    val W_a = ArrayBuffer[DenseMatrix[Double]]()
    val vW_a = ArrayBuffer[DenseMatrix[Double]]()

    for(h <- 0 to bc_parameter.value.layer - 2){
      val nndwi = train_nnbp(h)

      val nndwi2 = nndwi :* bc_parameter.value.alpha

      val nndwi3 = if(bc_parameter.value.momentum > 0 ){
        val vwi = bc_nn_vW.value(h)
        val dw3 = nndwi2 + (vwi * bc_parameter.value.momentum)
        dw3
      }else{
        nndwi2
      }
      W_a += (bc_nn_W.value(h) - nndwi3)

      val nnvwi1 = if (bc_parameter.value.momentum > 0) {
        val vwi = bc_nn_vW.value(h)
        val vw3 = nndwi2 + (vwi * bc_parameter.value.momentum)
        vw3
      } else {
        bc_nn_vW.value(h)
      }
      vW_a += nnvwi1
    }
    Array(W_a.toArray, vW_a.toArray)
  }
}
