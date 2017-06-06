package CNN

import java.awt.image.Kernel

import breeze.linalg.{*, DenseMatrix, kron, rot90, sum}
import breeze.numerics.{exp, sqrt, tanh}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.math._

/**
  * Created by root on 17-3-29.
  */

case class CNNLayers(
                      types: String,
                      outputmaps: Double,
                      kernelsize: Double,
                      scale: Double,
                      k: Array[Array[DenseMatrix[Double]]],
                      b: Array[Double],
                      dk: Array[Array[DenseMatrix[Double]]],
                      db: Array[Double]
                    ) extends Serializable
class CNN(
         private var inputsize: DenseMatrix[Double],
         private var architecture: Array[String],
         private var layer: Int,
         private var outsizelist: Array[Double],
         private var outdimension: Int,
         private var kernelsizelist: Array[Double],
         private var scales: Array[Double],
         private var alpha: Double
         ) extends Serializable{
  def this() = this(
    new DenseMatrix[Double](1, 2, Array(28.0, 28.0)),
    Array("i", "c", "s", "c", "s"),
    5,
    Array(0.0, 6.0, 0.0, 12.0, 0.0),
    10,
    Array(0.0, 5.0, 0.0, 5.0, 0.0),
    Array(0.0, 0.0, 2.0, 0.0, 2.0),
    1.0
  )

  def setInputSize(sizemap: DenseMatrix[Double]): this.type ={
    this.inputsize = sizemap
    this
  }

  def setArchitecture(architecture: Array[String]): this.type ={
    this.architecture = architecture
    this
  }

  def setLayer(layer: Int): this.type ={
    this.layer = layer
    this
  }

  def setOutSizeList(outsizelist: Array[Double]): this.type = {
    this.outsizelist = outsizelist
    this
  }

  def setOutDimension(outdimension: Int): this.type ={
    this.outdimension = outdimension
    this
  }

  def setKernelSizeList(kernelsizelist: Array[Double]): this.type ={
    this.kernelsizelist = kernelsizelist
    this
  }

  def setScales(scales: Array[Double]): this.type ={
    this.scales = scales
    this
  }

  def setAlpha(alpha: Double): this.type ={
    this.alpha = alpha
    this
  }

  private def CNNSetup(): (Array[CNNLayers], DenseMatrix[Double], DenseMatrix[Double], Double) ={
    var layerlist = ArrayBuffer[CNNLayers]()
    var tmpinputsize = 1.0
    var inputmap = inputsize

    for(l <- 0 to layer - 1){
      val layertype = architecture(l)
      val outputsize = outsizelist(l)
      val kernelsize =kernelsizelist(l)
      val scale = scales(l)

      val info = if(layertype == "c"){
        inputmap = inputmap - kernelsize + 1.0

        var ki = ArrayBuffer[Array[DenseMatrix[Double]]]()
        for(i <- 0 to tmpinputsize.toInt){
          val kj = ArrayBuffer[DenseMatrix[Double]]()
          for(j <- 0 to outputsize.toInt){
            val k = (DenseMatrix.rand[Double](kernelsize.toInt, kernelsize.toInt) - 0.5) * 2.0 * sqrt(6.0 / ((tmpinputsize * math.pow(kernelsize, 2)) + (outputsize * math.pow(kernelsize, 2))))
            kj += k
          }
          ki += kj.toArray
        }
        val b1 = Array.fill(outputsize.toInt)(0.0)
        tmpinputsize = outputsize

        new CNNLayers(layertype, outputsize, kernelsize, scale, ki.toArray, b1, ki.toArray, b1)
      }else if(layertype == "s"){
        inputmap = inputmap / scale
        val b1 = Array.fill(tmpinputsize.toInt)(0.0)
        val ki = Array(Array(DenseMatrix.zeros[Double](1, 1)))

        new CNNLayers(layertype, outputsize, kernelsize, scale, ki, b1, ki, b1)
      }else{
        val ki = Array(Array(DenseMatrix.zeros[Double](1, 1)))
        val b1 = Array(0.0)
        new CNNLayers(layertype, outputsize, kernelsize, scale, ki, b1, ki, b1)
      }
      layerlist += info
    }

    val finalfeaturesdimension = tmpinputsize * inputmap(0, 0) * inputmap(0, 1)
    val finalbias = DenseMatrix.zeros[Double](outdimension, 1)
    val finalweight = (DenseMatrix.rand[Double](outdimension, finalfeaturesdimension.toInt) - 0.5) * 2.0 * sqrt(6.0 / (outdimension + finalfeaturesdimension))
    (layerlist.toArray, finalbias, finalweight, alpha)
  }

  def train(train_d: RDD[(DenseMatrix[Double], DenseMatrix[Double])], batchSize: Int, iterate: Int): CNNModel ={
    val sc = train_d.sparkContext

    var (cnnlayers, cnnfb, cnnfw, ratealpha) = CNNSetup()

    val m = train_d.count
    val batchNum = (m / batchSize).toInt

    for(i <- 1 to iterate){
      val splitRate = Array.fill(batchNum)(1.0 / batchNum)
      for (b <- 1 to batchNum){
        val bc_cnnlayers = sc.broadcast(cnnlayers)
        val bc_cnnfb = sc.broadcast(cnnfb)
        val bc_cnnfw = sc.broadcast(cnnfw)
        val train_split2 = train_d.randomSplit(splitRate, System.nanoTime())
        val batch_xy1 = train_split2(b - 1)

        val train_cnnff = CNNExt.CNNff(batch_xy1,bc_cnnlayers, bc_cnnfb, bc_cnnfw)

        val train_cnnbp = CNNExt.CNNbp(train_cnnff, bc_cnnlayers, bc_cnnfb, bc_cnnfw)

        val newWeightAndKs = CNNExt.CNNUpdateWeightsAndKs(train_cnnbp, bc_cnnfb, bc_cnnfw, alpha)

        cnnfw = newWeightAndKs._1
        cnnfb = newWeightAndKs._2
        cnnlayers = newWeightAndKs._3
      }
    }
    new CNNModel(cnnlayers, cnnfw, cnnfb)
  }
}

object CNNExt extends Serializable{
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

  /**
    * 利用克罗内克积进行上采样
    *
    */

  def expand(a: DenseMatrix[Double], s: Array[Int]): DenseMatrix[Double] ={
    val b = DenseMatrix.ones[Double](s(0), s(1))
    val res = kron(a, b)
    res
  }

  def convn(m0: DenseMatrix[Double], k0: DenseMatrix[Double], shape: String): DenseMatrix[Double] = {
    //val m0 = BDM((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, 0.0), (0.0, 1.0, 1.0, 0.0))
    //val k0 = BDM((1.0, 1.0), (0.0, 1.0))
    //val m0 = BDM((1.0, 5.0, 9.0), (3.0, 6.0, 12.0), (7.0, 2.0, 11.0))
    //val k0 = BDM((1.0, 2.0, 0.0), (0.0, 5.0, 6.0), (7.0, 0.0, 9.0))
    val out1 = shape match {
      case "valid" =>
        val m1 = m0
        val k1 = rot90(rot90(k0))
        val row1 = m1.rows - k1.rows + 1
        val col1 = m1.cols - k1.cols + 1
        var m2 = DenseMatrix.zeros[Double](row1, col1)
        for (i <- 0 to row1 - 1) {
          for (j <- 0 to col1 - 1) {
            val r1 = i
            val r2 = r1 + k1.rows - 1
            val c1 = j
            val c2 = c1 + k1.cols - 1
            val mi = m1(r1 to r2, c1 to c2)
            m2(i, j) = (mi :* k1).sum
          }
        }
        m2
      case "full" =>
        var m1 = DenseMatrix.zeros[Double](m0.rows + 2 * (k0.rows - 1), m0.cols + 2 * (k0.cols - 1))
        for (i <- 0 to m0.rows - 1) {
          for (j <- 0 to m0.cols - 1) {
            m1((k0.rows - 1) + i, (k0.cols - 1) + j) = m0(i, j)
          }
        }
        val k1 = rot90(rot90(k0))
        val row1 = m1.rows - k1.rows + 1
        val col1 = m1.cols - k1.cols + 1
        var m2 = DenseMatrix.zeros[Double](row1, col1)
        for (i <- 0 to row1 - 1) {
          for (j <- 0 to col1 - 1) {
            val r1 = i
            val r2 = r1 + k1.rows - 1
            val c1 = j
            val c2 = c1 + k1.cols - 1
            val mi = m1(r1 to r2, c1 to c2)
            m2(i, j) = (mi :* k1).sum
          }
        }
        m2
    }
    out1
  }

  def CNNff(
             batch_xy1: RDD[(DenseMatrix[Double], DenseMatrix[Double])],
             bc_cnn_layers: org.apache.spark.broadcast.Broadcast[Array[CNNLayers]],
             bc_cnn_ffb: org.apache.spark.broadcast.Broadcast[DenseMatrix[Double]],
             bc_cnn_ffW: org.apache.spark.broadcast.Broadcast[DenseMatrix[Double]]): RDD[(DenseMatrix[Double], Array[Array[DenseMatrix[Double]]], DenseMatrix[Double], DenseMatrix[Double])] = {

    val train_1 = batch_xy1.map(b =>{
      val lable = b._1
      val featrues = b._2

      val nna1 = Array(featrues)

      val nna = ArrayBuffer[Array[DenseMatrix[Double]]]()

      nna += nna1
      (lable, nna)
    })

    val train_layers = train_1.map(t => {
      val lable = t._1
      val nna = t._2

      var inputsize = 1.0

      for(l <- 1 to bc_cnn_layers.value.length - 1){
        val layertype = bc_cnn_layers.value(l).types
        val scale = bc_cnn_layers.value(l).scale
        val outputsize = bc_cnn_layers.value(l).outputmaps
        val nnasubs = ArrayBuffer[DenseMatrix[Double]]()

        if(layertype == "c"){
          val kernelsize = bc_cnn_layers.value(l).kernelsize
          val kmatrix = bc_cnn_layers.value(l).k
          val bmatrix = bc_cnn_layers.value(l).b

          for(j <- 0 to outputsize.toInt - 1){
            var tmp = DenseMatrix.zeros[Double](nna(l - 1)(0).rows - kernelsize.toInt + 1, nna(l - 1)(0).cols - kernelsize.toInt + 1)
            for(i <- 0 to inputsize.toInt - 1){
              tmp = tmp + convn(nna(l - 1)(i), kmatrix(i)(j), "valid")
            }
            val nnasub = sigm(tmp + bmatrix(j))
            nnasubs += nnasub
          }
          nna += nnasubs.toArray
          inputsize = outputsize
        }else if(layertype == "s"){
          for(j <- 0 to inputsize.toInt - 1){
            val tmp = convn(nna(l - 1)(j), DenseMatrix.ones[Double](scale.toInt, scale.toInt) / (scale * scale), "valid")

            val tmp1 = tmp(::, 0 to -1 by scale.toInt).t + 0.0
            val tmp2 = tmp1(::, 0 to -1 by scale.toInt).t + 0.0
            nnasubs += tmp2
          }
          //println(nnasubs.length)
          nna += nnasubs.toArray
        }
      }

      val lastlayerfeatureslist = ArrayBuffer[Double]()

      for(i <- 0 to nna(bc_cnn_layers.value.length - 1).length - 1){
        lastlayerfeatureslist ++= nna(bc_cnn_layers.value.length - 1)(i).data
      }

      val lastlayerfeatures = new DenseMatrix[Double](lastlayerfeatureslist.length, 1, lastlayerfeatureslist.toArray)

      val nno = sigm(bc_cnn_ffW.value * lastlayerfeatures + bc_cnn_ffb.value)

      (lable, nna.toArray, lastlayerfeatures, nno)
    })
    train_layers
  }

  def CNNbp(
             train_cnnff: RDD[(DenseMatrix[Double], Array[Array[DenseMatrix[Double]]], DenseMatrix[Double], DenseMatrix[Double])],
             bc_cnn_layers: org.apache.spark.broadcast.Broadcast[Array[CNNLayers]],
             bc_cnn_ffb: org.apache.spark.broadcast.Broadcast[DenseMatrix[Double]],
             bc_cnn_ffW: org.apache.spark.broadcast.Broadcast[DenseMatrix[Double]]): (RDD[(DenseMatrix[Double], Array[Array[DenseMatrix[Double]]], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], Array[Array[DenseMatrix[Double]]])], DenseMatrix[Double], DenseMatrix[Double], Array[CNNLayers]) = {

    val n = bc_cnn_layers.value.length
    val train_lasterr = train_cnnff.map(t => {
      val nne = t._4 - t._1
      (t._1, t._2, t._3, t._4, nne)
    })

    val train_lastdelta = train_lasterr.map(t => {
      val nne = t._5
      val nno = t._4
      val lastlayerfeatures = t._3

      val nno_delta = nne :* (nno :* (1.0 - nno))

      val lastlayer_delta = if(bc_cnn_layers.value(n - 1).types == "c"){
        val lastfeature_delta = bc_cnn_ffW.value.t * nno_delta
        val lastC_delta = lastfeature_delta :* (lastlayerfeatures :* (1.0 - lastlayerfeatures))
        lastC_delta
      }else{
        val lastC_delta = bc_cnn_ffW.value.t * nno_delta
        lastC_delta
      }
      (t._1, t._2, t._3, t._4, t._5, nno_delta, lastlayer_delta)
    })

    val lastfeature_row = train_lastdelta.map(t => t._2(n - 1)(1)).take(1)(0).rows
    val lastfeature_clo = train_lastdelta.map(t => t._2(n - 1)(1)).take(1)(0).cols

    val lengthpermatrix = lastfeature_row * lastfeature_clo
    val train_alldelta = train_lastdelta.map(t => {
      val nna = t._2
      val lastlayer_delta = t._7

      var nndelta = new Array[Array[DenseMatrix[Double]]](n)
      val lastdelta = ArrayBuffer[DenseMatrix[Double]]()

      for(i <- 0 to nna(n - 1).length - 1){
        val tmp1 = lastlayer_delta((i * lengthpermatrix) to ((i + 1) * lengthpermatrix - 1), 0)
        val tmp2 = new DenseMatrix(lastfeature_row, lastfeature_clo, tmp1.toArray)
        lastdelta += tmp2
      }

      nndelta(n - 1) = lastdelta.toArray

      for(l <- (n - 2) to 0 by -1){
        val layertype = bc_cnn_layers.value(l).types
        val deltalist = ArrayBuffer[DenseMatrix[Double]]()

        if(layertype == "c"){
          for(i <- 0 to nna(l).length - 1){
            val tmp_a = nna(l)(i)
            val tmp_d = nndelta(l + 1)(i)
            val tmp_scale = bc_cnn_layers.value(l + 1).scale

            val tmp1 = tmp_a :* (1.0 - tmp_a)

            val tmp2 = expand(tmp_d, Array(tmp_scale.toInt, tmp_scale.toInt)) / (tmp_scale.toDouble * tmp_scale)

            deltalist += (tmp1 :* tmp2)
          }
        }else{
          for(i <- 0 to nna(l).length - 1){
            var z = DenseMatrix.zeros[Double](nna(l)(0).rows, nna(l)(0).cols)
            for(j <- 0 to nna(l + 1).length - 1){
              z = z + convn(nndelta(l + 1)(j), rot90(rot90(bc_cnn_layers.value(l + 1).k(i)(j))), "full")
            }
            deltalist += z
          }
        }
        nndelta(l) = deltalist.toArray
      }
      (t._1, t._2, t._3, t._4, t._5, t._6, t._7, nndelta)
    })

    var layers = bc_cnn_layers.value

    for(l <- 1 to layers.length - 1){
      val layertype = layers(l).types
      val currfeaturelen = train_alldelta.map(t => t._2(l).length).take(1)(0)
      val forwardfeaturelen = train_alldelta.map(t => t._2(l - 1).length).take(1)(0)
      if(layertype == "c"){
        var nndk = new Array[Array[DenseMatrix[Double]]](forwardfeaturelen)
        for (i <- 0 to forwardfeaturelen - 1) {
          for (j <- 0 to currfeaturelen - 1) {
            nndk(i) = new Array[DenseMatrix[Double]](currfeaturelen)
          }
        }
        var nndb = new Array[Double](currfeaturelen)
        for(j <- 0 to currfeaturelen - 1){
          for(i <- 0 to forwardfeaturelen - 1){
            val rdd_dk_ij = train_alldelta.map(t => {
              val nna = t._2
              val nnd = t._8

              val tmp_a = nna(l - 1)(i)
              val tmp_d = nnd(l)(j)
              convn(rot90(rot90(tmp_a)), tmp_d, "valid")
            })

            val initdk = DenseMatrix.zeros[Double](rdd_dk_ij.take(1)(0).rows, rdd_dk_ij.take(1)(0).cols)

            val (dk_ij, count_dk) = rdd_dk_ij.treeAggregate((initdk, 0L))(
              seqOp = (c, v) => {
                // c: (m, count), v: (m)
                val m1 = c._1
                val m2 = m1 + v
                (m2, c._2 + 1)
              },
              combOp = (c1, c2) => {
                // c: (m, count)
                val m1 = c1._1
                val m2 = c2._1
                val m3 = m1 + m2
                (m3, c1._2 + c2._2)
              })
            val dk = dk_ij / count_dk.toDouble
            nndk(i)(j) = dk
          }
          val rdd_db_j = train_alldelta.map(t => {
            val nnd = t._8
            val tmp_d = nnd(l)(j)
            sum(tmp_d)
          })

          val db_j = rdd_db_j.reduce(_ + _)
          val count_db = rdd_db_j.count
          val db = db_j / count_db.toDouble
          nndb(j) = db
        }
        layers(l) = new CNNLayers(layers(l).types, layers(l).outputmaps, layers(l).kernelsize, layers(l).scale, layers(l).k, layers(l).b, nndk, nndb)
      }
    }

    val train_dWeight = train_alldelta.map(t => {
      val nnod = t._6
      val lastfeatures = t._3

      nnod * lastfeatures.t
    })

    val train_dBias = train_alldelta.map(t => t._6)

    val initdWeight = DenseMatrix.zeros[Double](bc_cnn_ffW.value.rows, bc_cnn_ffW.value.cols)

    val (dweight, dweightcount) = train_dWeight.treeAggregate((initdWeight, 0L))(
      seqOp = (c, v) => {
        val bc = c._2
        val bv = c._1
        (bv + v, bc + 1)
      },
      combOp = (c1, c2) =>{
        (c1._1 + c2._1, c1._2 + c2._2)
      }
    )

    val nndweight = dweight / dweightcount.toDouble

    val initdBias = DenseMatrix.zeros[Double](bc_cnn_ffb.value.rows, bc_cnn_ffb.value.cols)

    val (dbias, dbiascount) = train_dBias.treeAggregate((initdBias, 0L))(
      seqOp = (c, v) =>{
        val bc = c._2
        val bv = c._1
        (bv + v, bc + 1)
      },
      combOp = (c1, c2) => {
        (c1._1 + c2._1, c1._2 + c2._2)
      }
    )

    val nndbias = dbias / dbiascount.toDouble

    (train_alldelta, nndweight, nndbias, layers)
  }

  def CNNUpdateWeightsAndKs(
                             train_cnnbp: (RDD[(DenseMatrix[Double], Array[Array[DenseMatrix[Double]]], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], Array[Array[DenseMatrix[Double]]])], DenseMatrix[Double], DenseMatrix[Double], Array[CNNLayers]),
                             bc_cnn_ffb: org.apache.spark.broadcast.Broadcast[DenseMatrix[Double]],
                             bc_cnn_ffW: org.apache.spark.broadcast.Broadcast[DenseMatrix[Double]],
                             alpha: Double
                           ): (DenseMatrix[Double], DenseMatrix[Double], Array[CNNLayers]) ={
    val train_alldelta = train_cnnbp._1
    val nndweight = train_cnnbp._2
    val nndbias = train_cnnbp._3
    var layers = train_cnnbp._4

    var nnWeight = bc_cnn_ffW.value
    var nnBias = bc_cnn_ffb.value

    for(l <- 1 to layers.length - 1) {
      val layertype = layers(l).types
      val currfeaturelen = train_alldelta.map(t => t._2(l).length).take(1)(0)
      val forwardfeaturelen = train_alldelta.map(t => t._2(l - 1).length).take(1)(0)
      if(layertype == "c"){
        for(j <- 0 to currfeaturelen - 1){
          for(i <- 0 to forwardfeaturelen - 1){
            layers(l).k(i)(j) = layers(l).k(i)(j) - layers(l).dk(i)(j) * alpha
          }
          layers(l).b(j) = layers(l).b(j) - layers(l).db(j) * alpha
        }
      }
    }
    println(nndweight.rows + "**" + nndweight.cols + "~" + nnWeight.rows + "**" + nnWeight.cols)
    nnWeight = nnWeight - nndweight * alpha
    nnBias = nnBias - nndbias * alpha

    (nnWeight, nnBias, layers)
  }
}
