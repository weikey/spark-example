package org.weikey.cf

/**
 * Created with IntelliJ IDEA.
 * User: weikey
 * Date: 13-12-9
 * Time: 下午11:36
 * To change this template use File | Settings | File Templates.
 */

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import scala.math.sqrt
import org.apache.spark.rdd.RDD

object ItemBased2 {

  def correlation(size: Double, dotProduct: Double, ratingSum: Double,
                  rating2Sum: Double, ratingNormSq: Double, rating2NormSq: Double) = {

    val numerator = size * dotProduct - ratingSum * rating2Sum
    val denominator = math.sqrt(size * ratingNormSq - ratingSum * ratingSum) * math.sqrt(size * rating2NormSq - rating2Sum * rating2Sum)

    numerator / denominator
  }

  def regularizedCorrelation(size: Double, dotProduct: Double, ratingSum: Double,
                             rating2Sum: Double, ratingNormSq: Double, rating2NormSq: Double,
                             virtualCount: Double, priorCorrelation: Double) = {

    val unregularizedCorrelation = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
    val w = size / (size + virtualCount)

    w * unregularizedCorrelation + (1 - w) * priorCorrelation
  }

  def cosineSimilarity(dotProduct: Double, ratingNorm: Double, rating2Norm: Double) = {
    dotProduct / (ratingNorm * rating2Norm)
  }

  def jaccardSimilarity(usersInCommon: Double, totalUsers1: Double, totalUsers2: Double) = {
    val union = totalUsers1 + totalUsers2 - usersInCommon
    usersInCommon / union
  }

  def euclideanDistance(distance: Double)={
    (1 / (1 + sqrt(distance)))
  }

  def recommender(similarities: RDD[((Int,Int),Double)] ,ratings: RDD[(Int,Int,Double)],recommendItemNum: Int)={

    val corr = similarities.map {
      p =>
        List((p._1._1, (p._1._2, p._2)), (p._1._2, (p._1._1, p._2)), (p._1._1, (p._1._1, 1.0)), (p._1._2, (p._1._2, 1.0)))
    }.flatMap(q => q.toList).distinct().groupByKey().map {
      x =>
        val sortArray = x._2.sortWith(_._1 < _._1) //排序
        (x._1, sortArray)
    }
    //    println("similarities: ")
    //    corr.foreach(println)

    val userPrefItemSimilarity = ratings.groupBy(x => x._2).join(corr).map {
      y =>
        y._2._1.map {
          z => (z._1, (z._3, y._2._2
            ))
        }
    }.flatMap(p => p.toList)
    //      .groupByKey()
    //.foreach(println)

//    var recommendItemNum = 4
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

  }

  def main(args: Array[String]) {
    val TRAIN_FILENAME = args(1)

    // Spark programs require a SparkContext to be initialized
    //val sc = new SparkContext(args(0), "MovieSimilarities")
    val sc = new SparkContext(args(0), "MovieSimilarities", System.getenv("SPARK_HOME"), Seq(System.getenv("SPARK_EXAMPLES_JAR")))
    // extract (userid, movieid, rating) from ratings data
    val ratings = sc.textFile(TRAIN_FILENAME)
      .map(line => {
      val fields = line.split(args(2))
      (fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    })

    // get num raters per movie, keyed on movie id
    val numRatersPerMovie = ratings
      .groupBy(tup => tup._2)
      .map(grouped => (grouped._1, grouped._2.size))

    // join ratings with num raters on movie id
    val ratingsWithSize = ratings
      .groupBy(tup => tup._2)
      .join(numRatersPerMovie)
      .flatMap(joined => {
      joined._2._1.map(f => (f._1, f._2, f._3, joined._2._2))
    })

    // ratingsWithSize now contains the following fields: (user, movie, rating, numRaters).

    // dummy copy of ratings for self join
    val ratings2 = ratingsWithSize.keyBy(tup => tup._1)

    // join on userid and filter movie pairs such that we don't double-count and exclude self-pairs
    val ratingPairs =
      ratingsWithSize
        .keyBy(tup => tup._1)
        .join(ratings2)
        .filter(f => f._2._1._2 < f._2._2._2)

    // compute raw inputs to similarity metrics for each movie pair
    val vectorCalcs =
      ratingPairs
        .map(data => {
        val key = (data._2._1._2, data._2._2._2)
        val stats =
          (data._2._1._3 * data._2._2._3, // rating 1 * rating 2
            data._2._1._3, // rating movie 1
            data._2._2._3, // rating movie 2
            math.pow(data._2._1._3, 2), // square of rating movie 1
            math.pow(data._2._2._3, 2), // square of rating movie 2
            data._2._1._4, // number of raters movie 1
            data._2._2._4, // number of raters movie 2
            math.pow(data._2._1._3 - data._2._2._3, 2)) // square of (rating 1 - rating 2)
        (key, stats)
      })
        .groupByKey()
        .map(data => {
        val key = data._1
        val vals = data._2
        val size = vals.size
        val dotProduct = vals.map(f => f._1).sum
        val ratingSum = vals.map(f => f._2).sum
        val rating2Sum = vals.map(f => f._3).sum
        val ratingSq = vals.map(f => f._4).sum
        val rating2Sq = vals.map(f => f._5).sum
        val numRaters = vals.map(f => f._6).max
        val numRaters2 = vals.map(f => f._7).max
        val ratingRemainder2SqSum = vals.map(f => f._8).sum
        (key, (size, dotProduct, ratingSum, rating2Sum, ratingSq, rating2Sq, numRaters, numRaters2,ratingRemainder2SqSum))
      })

    val PRIOR_COUNT = 10
    val PRIOR_CORRELATION = 0

    // compute similarity metrics for each movie pair
    val similarities =
      vectorCalcs
        .map(fields => {

        val key = fields._1
        val (size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq, numRaters, numRaters2,ratingRemainder2SqSum) = fields._2

        val corr = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
        val regCorr = regularizedCorrelation(size, dotProduct, ratingSum, rating2Sum,
          ratingNormSq, rating2NormSq, PRIOR_COUNT, PRIOR_CORRELATION)
        val cosSim = cosineSimilarity(dotProduct, scala.math.sqrt(ratingNormSq), scala.math.sqrt(rating2NormSq))
        val jaccard = jaccardSimilarity(size, numRaters, numRaters2)
        val euclidean = euclideanDistance(ratingRemainder2SqSum)
//        (corr, regCorr, cosSim, jaccard,ratingRemainder2SqSum,euclidean)
        (key, corr)
      })
//        .take(10).foreach(println)
    recommender(similarities,ratings,4)
    //      val  corr=similarities.map(p=>
    //        ((p._1._1,(p._1._2,p._2)),(p._1._2,(p._1._1,p._2)),(p._1._1,(p._1._1,1.0)),(p._1._2,(p._1._2,1.0)))
    //        ).flatMap(q=>List(q._1,q._2,q._3,q._4)).distinct().groupByKey().map {
    //        x =>
    //          val sortArray = x._2.sortWith(_._1 < _._1) //排序
    //          (x._1, sortArray)
    //      }

  }

}
