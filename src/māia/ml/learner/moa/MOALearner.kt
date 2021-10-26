/*
 * MOALearner.kt
 * Copyright (C) 2021 University of Waikato, Hamilton, New Zealand
 *
 * This file is part of MĀIA.
 *
 * MĀIA is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MĀIA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MĀIA.  If not, see <https://www.gnu.org/licenses/>.
 */
package māia.ml.learner.moa

import com.yahoo.labs.samoa.instances.*
import moa.classifiers.MultiClassClassifier
import moa.core.Example
import moa.learners.Learner
import māia.ml.dataset.DataRow
import māia.ml.dataset.DataStream
import māia.ml.dataset.headers.DataColumnHeaders
import māia.ml.dataset.headers.ensureOwnership
import māia.ml.dataset.headers.viewColumns
import māia.ml.dataset.moa.dataRowToInstanceExample
import māia.ml.dataset.moa.representationParseMOAValue
import māia.ml.dataset.moa.withColumnHeadersToInstancesHeader
import māia.ml.dataset.type.DataRepresentation
import māia.ml.learner.AbstractLearner
import māia.ml.learner.type.Classifier
import māia.ml.learner.type.LearnerType
import māia.ml.learner.type.SingleTarget
import māia.ml.learner.type.intersectionOf
import māia.util.maxWithIndex

val MOA_LEARNER_TYPE = intersectionOf(SingleTarget, Classifier)

/**
 * TODO
 */
class MOALearner(
    val source : Learner<Example<Instance>>
) : AbstractLearner<DataStream<*>>(
    MOA_LEARNER_TYPE,
    DataStream::class
) {
    init {
        if (source !is MultiClassClassifier)
            throw Exception("source is not a MultiClassClassifier")
    }

    lateinit var instancesHeader : InstancesHeader

    override fun performInitialisation(
        headers : DataColumnHeaders
    ) : Triple<DataColumnHeaders, DataColumnHeaders, LearnerType> {
        val predictInputHeaders = headers.viewColumns { _, header -> !header.isTarget}
        val predictOutputHeaders = headers.viewColumns { _, header -> header.isTarget}

        instancesHeader = withColumnHeadersToInstancesHeader(headers)

        source.modelContext = instancesHeader

        return Triple(
            predictInputHeaders,
            predictOutputHeaders,
            MOA_LEARNER_TYPE
        )
    }

    override fun performTrain(
        trainingDataset : DataStream<*>
    ) {
        for (row in trainingDataset.rowIterator())
            source.trainOnInstance(
                dataRowToInstanceExample(row, instancesHeader)
            )
    }

    override fun performPredict(
        row : DataRow
    ) : DataRow {
        val instance = dataRowToInstanceExample(row, instancesHeader, true)

        val votes = source.getVotesForInstance(instance)

        val prediction = votes.iterator().maxWithIndex().first.toDouble()

        return object : DataRow {
            override val headers : DataColumnHeaders = this@MOALearner.predictOutputHeaders
            override fun <T> getValue(
                representation : DataRepresentation<*, *, out T>
            ) : T  = this@MOALearner.predictOutputHeaders.ensureOwnership(representation) {
                representationParseMOAValue(this, prediction)
            }
        }
    }

}
