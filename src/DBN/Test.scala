package DBN

import breeze.linalg.DenseMatrix
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 17-3-20.
  */
object Test {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("NetTest").setMaster("local[4]").set("spark.executor.memory", "8g")
    val sc = new SparkContext(conf)
    DBNTest(sc)
  }
  def DBNTest(sc: SparkContext): Unit ={
    Logger.getRootLogger.setLevel(Level.WARN)
    val data_path = "/root/IdeaProjects/SparkMLlibDeepLearn/src/Data/data1.txt"
    val examples = sc.textFile(data_path).cache()
    val train_d1 = examples.map { line =>
      val f1 = line.split("\t")
      val f = f1.map(f => f.toDouble)
      //val id = f(0)
      val y = Array(f(0))
      val x = f.slice(1, f.length)
      (1, new DenseMatrix(1, y.length, y), new DenseMatrix(1, x.length, x))
    }
    val train_d = train_d1.map(f => (f._2, f._3))
    val opts = Array(100.0, 20.0, 0.0)

    //3 设置训练参数，建立DBN模型
    val DBNmodel = new DBN().
      setSize(Array(5, 7)).
      setLayer(2).
      setMomentum(0.1).
      setAlpha(1.0).DBNTrain(train_d, 100, 20)

    DBNmodel.printInfo()
  }
}

