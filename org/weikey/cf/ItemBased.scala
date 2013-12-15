package org.weikey.cf

/**
 * Created with IntelliJ IDEA.
 * User: weikey
 * Date: 13-12-7
 * Time: 下午7:50
 * To change this template use File | Settings | File Templates.
 */

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import scala.math.sqrt
import scala.collection.immutable.VectorBuilder
import cern.jet.math._
import cern.colt.matrix._
import cern.colt.matrix.linalg._
import scala._
import scala.util.Random

object ItemBased {

  def Euclidean_Distance(v1: Seq[(Long, Double)], v2: Seq[(Long, Double)]): Double = {
    var distance = 0.0
    var sameCount = 0
    for (
      a <- v1; b <- v2 if (a._1 == b._1)
    ) {
      sameCount += 1
      distance += (a._2 - b._2) * (a._2 - b._2)
    }
    distance = if (sameCount == 0) 0.0 else (1 / (1 + sqrt(distance)))
    distance
  }

  //Corr(X,Y)=n∑xy−∑x∑y/(√(n∑x2−(∑x)2)√(n∑y2−(∑y)2))
  def Pearson_Correlation_Coefficient2(v1: Seq[(Long, Double)], v2: Seq[(Long, Double)]): Double = {
    var distance = 0.0
    var sameCount = 0
    var a_sum = 0.0
    var b_sum = 0.0
    var ab_sum = 0.0
    var aa_sum = 0.0
    var bb_sum = 0.0
    for (
      a <- v1; b <- v2 if (a._1 == b._1)
    ) {
      sameCount += 1
      a_sum += a._2
      b_sum += b._2
      aa_sum += a._2 * a._2
      bb_sum += b._2 * b._2
      ab_sum += a._2 * b._2
    }
    distance = if (sameCount == 0) 0.0 else correlation(sameCount, ab_sum, a_sum, b_sum, aa_sum, bb_sum)
    distance
  }


  //Corr(X,Y)=n∑xy−∑x∑y/(√(n∑x2−(∑x)2)√(n∑y2−(∑y)2))
  def Pearson_Correlation_Coefficient(v1: Seq[(Long, Double)], v2: Seq[(Long, Double)]): Double = {
    var distance = 0.0
    var sameCount = 0
    var aav = 0.0
    var bbv = 0.0
    var abv_sum = 0.0
    var aav_sum = 0.0
    var bbv_sum = 0.0
    val a_avg=v1.map(_._2).sum/v1.map(_._2).length
    val b_avg=v2.map(_._2).sum/v2.map(_._2).length
    for (
      a <- v1; b <- v2 if (a._1 == b._1)
    ) {
      sameCount += 1
      aav = a._2-a_avg
      bbv = b._2-b_avg
      aav_sum += aav * aav
      bbv_sum += bbv * bbv
      abv_sum += aav * bbv
    }
    distance = if (sameCount == 0) 0.0 else  abv_sum/(math.sqrt(aav_sum)*math.sqrt(bbv_sum))
    distance
  }

  def correlation(size: Double, dotProduct: Double, ratingSum: Double,
                  rating2Sum: Double, ratingNormSq: Double, rating2NormSq: Double) = {

    val numerator = size * dotProduct - ratingSum * rating2Sum
    val denominator = math.sqrt(size * ratingNormSq - ratingSum * ratingSum) * math.sqrt(size * rating2NormSq - rating2Sum * rating2Sum)

    numerator / denominator
  }

  def Regularized_Pearson_Correlation_Coefficient(v1: Seq[(Long, Double)], v2: Seq[(Long, Double)]): Double = {
    val PRIOR_COUNT = 10
    val PRIOR_CORRELATION = 0
    var distance = 0.0
    var sameCount = 0
    var a_sum = 0.0
    var b_sum = 0.0
    var ab_sum = 0.0
    var aa_sum = 0.0
    var bb_sum = 0.0
    for (
      a <- v1; b <- v2 if (a._1 == b._1)
    ) {
      sameCount += 1
      a_sum += a._2
      b_sum += b._2
      aa_sum += a._2 * a._2
      bb_sum += b._2 * b._2
      ab_sum += a._2 * b._2
    }
    distance = if (sameCount == 0) 0.0 else regularized_correlation(sameCount, ab_sum, a_sum, b_sum, aa_sum, bb_sum, PRIOR_COUNT, PRIOR_CORRELATION)
    distance
  }

  def regularized_correlation(size: Double, dotProduct: Double, ratingSum: Double,
                              rating2Sum: Double, ratingNormSq: Double, rating2NormSq: Double,
                              virtualCount: Double, priorCorrelation: Double) = {

    val unregularizedCorrelation = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
    val w = size / (size + virtualCount)

    w * unregularizedCorrelation + (1 - w) * priorCorrelation
  }

  def cosine_similarity(v1: Seq[(Long, Double)], v2: Seq[(Long, Double)]): Double = {
    var distance = 0.0
    var sameCount = 0
    var ab_sum = 0.0
    var aa_sum = 0.0
    var bb_sum = 0.0
    for (
      a <- v1; b <- v2 if (a._1 == b._1)
    ) {
      sameCount += 1
      aa_sum += a._2 * a._2
      bb_sum += b._2 * b._2
      ab_sum += a._2 * b._2
    }
    distance = if (sameCount == 0) 0.0 else ab_sum / (aa_sum * bb_sum)
    distance
  }

  def jaccard_similarity(v1: Seq[(Long, Double)], v2: Seq[(Long, Double)]): Double = {
    var distance = 0.0
    var sameCount = 0
    for (
      a <- v1; b <- v2 if (a._1 == b._1)
    ) {
      sameCount += 1
    }

    distance = if (sameCount == 0) 0.0
    else {
      var union = v1.size + v2.size - sameCount
      sameCount / union
    }
    distance
  }

  //  def rmse(targetR: DoubleMatrix2D, ms: Array[DoubleMatrix1D],
  //           us: Array[DoubleMatrix1D]): Double =
  //  {
  //    val U = userVector.size
  //    val M = itemVector.size
  //    val r = factory2D.make(M, U)
  //    for (i <- 0 until M; j <- 0 until U) {
  //      r.set(i, j, blas.ddot(ms(i), us(j)))
  //    }
  //    //println("R: " + r)
  //      blas.daxpy(-1, targetR, r)
  //    val sumSqs = r.aggregate(Functions.plus, Functions.square)
  //    return sqrt(sumSqs / (M * U))
  //  }

  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "ItemCF", System.getenv("SPARK_HOME"), Seq(System.getenv("SPARK_EXAMPLES_JAR")))
    val lines = sc.textFile(args(0))
    val data = lines.map {
      line =>
        val vs = line.split("::")
        ((vs(0).toLong, (vs(1).toLong, vs(2).toDouble)), (vs(1).toLong, (vs(0).toLong, vs(2).toDouble)))
    }
    //      .filter{x=>if(Random.nextInt(10)<3) true else false}
    val userVector = data.map(_._1).groupByKey().filter(_._2.length > 1)
    val itemVector = sc.broadcast(data.map(_._2).groupByKey().collectAsMap())

    val itemPairs = userVector.map {
      p =>
        val is = for (
          item <- p._2; other <- p._2 if item._1 >= other._1
        ) yield (item._1, other._1)
        is
    }.flatMap(p => p)
      .distinct()

    val euclidean =
      itemPairs.map {
        p =>
          var distance = 1.0
          if (!p._1.equals(p._2)) {
            val iv1 = itemVector.value(p._1)
            val iv2 = itemVector.value(p._2)
            distance = Euclidean_Distance(iv1, iv2)
//            distance =Pearson_Correlation_Coefficient(iv1, iv2)
          }
          ((p._1, (p._2, distance)), (p._2, (p._1, distance)))
      }.flatMap(x => List(x._1, x._2)).distinct().groupByKey().map {
        x =>
          val sortArray = x._2.sortWith(_._1 < _._1) //排序
          (x._1, sortArray)
      }
//    euclidean.foreach(println)

    //outputformat:(userid,(pref,(similarityVect)))
    val userPrefItemSimilarity = euclidean.map {
      x =>
        (itemVector.value(x._1).map {
          y =>
            (y._1, (y._2, x._2))
        })
    }.flatMap(z => z.toList)
//          .foreach(println)

    var recommendItemNum = 4

    userPrefItemSimilarity.groupByKey().map {
      x =>
      // (pref,(similarityVect))
        val userid = x._1
        val abVect = x._2.map(y =>
        // similarityVect
          y._2.map(z => (z._2 * y._1, z._2))
        )
        val num = abVect.head.length
        var aSum = List.fill(num)(0.0)
        var bSum = List.fill(num)(0.0)
        for (ab <- abVect) {
          aSum = (aSum, ab.map {
            x => x._1
          }.toList).zipped.map(_ + _)
          bSum = (bSum, ab.map {
            x => x._2
          }.toList).zipped.map(_ + _)
        }
        val itemidList = x._2.head._2.map(y => y._1).toList
        //        val predictionList = (aSum, bSum).zipped.map(_ / _).toList
        //        (userid, (itemidList, predictionList).zipped.map((_, _)).sortWith(_._2 > _._2).take(recommendItemNum))

        (userid, (itemidList, aSum, bSum).zipped.map {
          case (a, b, c) => (a, b / c)
        }.sortWith(_._2 > _._2).take(recommendItemNum))
    }.foreach(println)


    //    val pearson = itemPairs.map { p =>
    //      val iv1 = itemVector.value(p._1)
    //      val iv2 = itemVector.value(p._2)
    //      var distance = Pearson_Correlation_Coefficient(iv1, iv2)
    //      (p._1, p._2, distance)
    //    }.toArray
    //
    //    val cosine = itemPairs.map { p =>
    //      val iv1 = itemVector.value(p._1)
    //      val iv2 = itemVector.value(p._2)
    //      var distance = cosine_similarity(iv1, iv2)
    //      (p._1, p._2, distance)
    //    }.toArray
    //
    //    val jaccard = itemPairs.map { p =>
    //      val iv1 = itemVector.value(p._1)
    //      val iv2 = itemVector.value(p._2)
    //      var distance = jaccard_similarity(iv1, iv2)
    //      (p._1, p._2, distance)
    //    }.toArray
    //
    //    println("euclidean: " + euclidean.mkString(" "))




    //    println("pearson: " + pearson.mkString(" "))
    //
    //    println("cosine: " + cosine.mkString(" "))
    //    println("jaccard: " + jaccard.mkString(" "))

  }

}
